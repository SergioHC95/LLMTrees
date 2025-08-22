#========================================================================================================
#========================================================================================================
#========================================================================================================

"""
Hierarchical Chunker class for GPT-4
"""


import openai
import json
import re
from typing import List, Dict, Tuple, Any, Optional
import tiktoken
from ast import literal_eval      # safer than json.loads for a tiny int list
import unicodedata
import concurrent.futures as cf
import threading
from tqdm import tqdm

class HierarchicalChunker:

    @staticmethod
    def strip_code_fences(segments: List[str]) -> List[str]:
        """
        Remove stray triple-backtick code-fence markers but KEEP any spaces
        that belong to the original text.

        • Strips an opening fence  ```[lang]   that appears at the very start.
        ─ Removes the backticks and optional language tag, **not** the first space.
        • Strips a closing fence   ```         that appears at the very end.
        ─ Removes the backticks (and any spaces after them), **not** the last space.
        """
        cleaned: List[str] = []
        for seg in segments:
            # opening fence: optional leading spaces, ``` and optional language tag
            seg = re.sub(r'^\s*```[\w+-]*', '', seg)

            # closing fence: ``` plus any trailing whitespace up to segment end
            seg = re.sub(r'```[\s]*$', '', seg)

            cleaned.append(seg)
        return cleaned    


    def __init__(self, api_key: str,
                 model: str = "gpt-4o",
                 *,
                 client: Optional[object] = None,
                 k: int = 4,
                 short_cutoff: int = 6,
                 max_workers: int = 8,
                 multithread: bool = False,        # ← NEW
                 debug: bool = False
                 ):
        """
        Initialize the hierarchical chunker.

        Args:
            api_key:       OpenAI API key
            model:         OpenAI model to use
            k:             Maximum number of segments at each level
            short_cutoff:  If the span has ≤ this many tokens,
                           use index-based segmentation
        """
        self.multithread  = multithread
        self.max_workers  = max_workers      # remember the desired pool size
        self.debug       = debug

        # stats + lock (needed for multithread mode)
        self.total_spans          = 0
        self.fallback_spans       = 0
        self.fallback_token_total = 0
        self.fallback_lengths: list[int] = []   # NEW – store every span length
        self.fallback_segments: list[List[str]] = []
        self.fallback_texts: list[str] = []      # NEW – store original text of fallback spans
        self._lock = threading.Lock()
        self._leaf_lock   = threading.Lock()
        self.leaf_count   = 0
        self.total_tokens = 0
        self.pbar: Optional[tqdm] = None

        if client is not None:
            self.client = client
        else:
            self.client = openai.OpenAI(api_key=api_key)

        self.model  = model
        self.k      = k
        self.encoding = tiktoken.get_encoding("cl100k_base")  # select the tokenizer based on model

        # --- new attribute ---
        self.short_cutoff = short_cutoff

    # ───────────────────────────────────────────────────────────────────
    def _call_llm(self,
                  system_prompt: str,
                  user_prompt: str,
                  *,
                  temperature: float = 0.1,
                  max_tokens: int = 2000,
                  model: Optional[str] = None) -> str:
        """
        Unified LLM wrapper.

        Returns
        -------
        str  – stripped content of the first assistant message.
        """
        chosen_model = model or self.model      # default to instance model

        response = self.client.chat.completions.create(
            model      = chosen_model,
            messages   = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature = temperature,
            max_tokens  = max_tokens,
        )
        return response.choices[0].message.content.strip()
    # ───────────────────────────────────────────────────────────────────

    def tokenize_text(self, text: str) -> Tuple[List[str], List[int]]:
        """
        Tokenise `text` and return:

            tokens        – list of strings that appear verbatim in `text`
            token_indices – their starting character indices in `text`

        Works with GPT-style BPE (OpenAI/tiktoken) **and** SentencePiece
        (Llama-2/3/4, DeepSeek).  The only adjustment is to translate the
        SentencePiece marker '▁' into an ordinary leading space.
        """
        token_ids = self.encoding.encode(text)

        tokens: List[str] = []
        indices: List[int] = []
        cur = 0
        sp_marker = "▁"           # U+2581 used by SentencePiece

        for tid in token_ids:
            tok = self.encoding.decode([tid])   # decode single token ID

            # SentencePiece: turn '▁word' into ' word'
            if tok.startswith(sp_marker):
                tok = " " + tok[1:]

            # Locate token in original text from current position
            idx = text.find(tok, cur)
            if idx == -1:                        # very rare fallback
                idx = cur
            tokens.append(tok)
            indices.append(idx)
            cur = idx + len(tok)

        if self.debug:
            print(f"[tokenize_text] → got {len(token_ids)} token_ids; first 10 IDs: {token_ids[:10]}")
            print(f"[tokenize_text] → decoded tokens: {tokens[:10]} … (total {len(tokens)})")

        return tokens, indices


    def get_system_prompt(self) -> str:
        """Get the system prompt for LLM segmentation."""
        return (
            "You are a text-segmentation assistant.\n"
            f"Split the entire input into up to *{self.k}* contiguous, non-overlapping segments such that concatenating them in order reproduces the original text exactly. Including ALL whitespaces and punctuation.\n"
            "Each segment should be semantically coherent.\n"
            "You should always return more than one segment, unless it is already a single token.\n"
            "In the case of simple phrases, segment into semantically contiguous chunks of tokens.\n"
            "*CRITICAL*: Preserve ALL whitespace and quotation marks exactly as it appears in the input. If the input starts with a space, the first segment should start with that space.\n"
            "Return a raw JSON array of strings — do NOT wrap in markdown or code fences.\n"
            "Examples:\n"
            "Input: \"The quick brown fox jumps over the lazy dog.\"\n"
            "Output: [\"The quick brown fox\", \" jumps over\", \" the lazy dog.\"]\n\n"
            "Input: \" brown fox\"\n"
            "Output: [\" brown\", \" fox\"]\n\n"
            "Input: \"the lazy dog.\"\n"
            "Output: [\"the\", \" lazy dog\", \".\"]\n\n"
            "Input: \" jumps over\"\n"
            "Output: [\" jumps\", \" over\"]\n\n"
        )

    def get_sentence_system_prompt(self) -> str:
        """
        Sentence-ID mode, but the model returns a list of CUT POINTS
        rather than groups.  A cut-point i (1 ≤ i < N) means
        “start a new segment **at sentence i**”.
        """
        return (
            "You are a text-segmentation assistant **in CUT-POINT mode**.\n"
            f"The input will be N numbered sentences (0 … N-1).\n"
            f"Choose up to *{self.k-1}* cut points so that the resulting segments\n"
            "are semantically contiguous and most conceptually disjoint from the other segments.\n"
            "Return **raw JSON** – a strictly ascending list of integers in the\n"
            "range 1 … N-1.  Do NOT wrap in markdown.\n"
            "Example (two segments):\n"
            "Input:\n"
            "[0] Yeah I was in the boy scouts at the time.\n"
            "[1] And we was doing the 50-yard dash racing but we was at the pier marked off and so we was doing the 50-yard dash.\n"
            "[2] There was about 8 or 9 of us you know, going down, coming back.\n"
            "[3] And going down the third time I caught cramps and I started yelling 'Help!' but the fellows didn't believe me you know.\n"
            "[4] They thought I was just trying to catch up because I was going on or slowing down.\n"
            "Output: [1, 3]\n"
        )
    # ❶  ────────────────────────────────────────────────────────────────────

    def reconcile_boundaries(self, original: str, segments: List[str]) -> List[str]:
        """
        Post-process *segments* so that their concatenation equals *original*
        even when the LLM dropped / duplicated leading spaces or quotes.

        Order of fixes
        ---------------
        1. leading whitespace
        2. ending quotation mark(s)
        3. existing length-based add / trim
        """

        # ---------------------------------------------------------------------
        # Character classes used throughout
        # ---------------------------------------------------------------------
        _WS          = r' \t'                      # leading whitespace
        _QUOTES      = r'"“”‘’\''                  # straight + curly + single
        _BOUNDARY_CH = _WS + _QUOTES

        _LEAD_WS_RE  = re.compile(rf'^([{_WS}]+)')          # run of leading spaces/tabs
        _TRAIL_Q_RE  = re.compile(rf'([{_QUOTES}]+\s*)$')   # quotes (opt. space) at end
        _LEAD_ANY_RE = re.compile(rf'^([{_BOUNDARY_CH}]+)')
        _TAIL_ANY_RE = re.compile(rf'([{_BOUNDARY_CH}]+)$')

        if not segments:
            return segments

        # ────────────────────────────────────────────────────────────────
        # 1️⃣  Leading whitespace  (spaces/tabs only)
        # ────────────────────────────────────────────────────────────────
        lead_ws_match = _LEAD_WS_RE.match(original)
        if lead_ws_match:
            lead_ws = lead_ws_match.group(0)
            if not segments[0].startswith(lead_ws):
                segments[0] = lead_ws + segments[0]

                if ''.join(segments) == original:
                    return segments

        # ────────────────────────────────────────────────────────────────
        # 2️⃣  Ending quotation mark(s) on the last segment
        #     (handles straight or curly quotes, possibly trailed by space)
        # ────────────────────────────────────────────────────────────────
        trail_q_match = _TRAIL_Q_RE.search(original)
        if trail_q_match:
            trail_q = trail_q_match.group(0)
            if not segments[-1].endswith(trail_q):
                # append the missing tail, one char at a time until it matches
                needed = trail_q
                while not segments[-1].endswith(trail_q) and needed:
                    segments[-1] += needed[0]
                    needed = needed[1:]

                if ''.join(segments) == original:
                    return segments

        # ────────────────────────────────────────────────────────────────
        # 3️⃣  Original length-based add / trim (your previous logic)
        # ────────────────────────────────────────────────────────────────
        _FRONT = _BOUNDARY_CH
        _BACK  = _FRONT[::-1]

        concat = ''.join(segments)
        if len(concat) < len(original):
            # -------- add missing on the left ----------
            lead_any = _LEAD_ANY_RE.match(original)
            if lead_any:
                lead = lead_any.group(0)
                while not segments[0].startswith(lead) and lead:
                    segments[0] = lead[-1] + segments[0]
                    lead = lead[:-1]

            # -------- add missing on the right ----------
            tail_any = _TAIL_ANY_RE.search(original)
            if tail_any:
                tail = tail_any.group(0)
                while not segments[-1].endswith(tail) and tail:
                    segments[-1] = segments[-1] + tail[0]
                    tail = tail[1:]

        # -------- trim surplus if now too long ----------
        concat = ''.join(segments)
        if len(concat) > len(original):
            surplus = len(concat) - len(original)
            while surplus and segments[0] and segments[0][0] in _FRONT:
                segments[0] = segments[0][1:]
                surplus -= 1
            while surplus and segments[-1] and segments[-1][-1] in _BACK:
                segments[-1] = segments[-1][-1:]
                surplus -= 1

        # Optional debug output
        if ''.join(segments) != original and getattr(self, "debug", False):
            print("[reconcile_boundaries] could not match bytes exactly")
            print(f"  original: '{original}'")
            print(f"  segments: {segments}")

        return segments


    def segment_text(self, text: str, temperature: float = 0.1, level: int = 0) -> List[str]:
        """
        Use LLM to segment text into chunks.
        
        Args:
            text: Text to segment
            temperature: OpenAI temperature parameter
            level: Current level for debugging
            
        Returns:
            List of text segments
        """
        system_prompt = self.get_system_prompt()
        user_prompt = f"```{text}```"
        
        if self.debug:
            print(f"\n{'='*60}")
            print(f"SEGMENTING AT LEVEL {level} (temp={temperature})")
            print(f"{'='*60}")
            print(f"Input text: '{text}'")
            # print(f"System prompt: {system_prompt}")
            # print(f"User prompt: {user_prompt}")
        
        try:
            content = self._call_llm(system_prompt,
                                    user_prompt,
                                    temperature=temperature)

            if self.debug:
                print(f"LLM Response: {content}")
            
            # Try to parse as JSON
            try:
                # segments = json.loads(content)
                # segments = self.strip_code_fences(segments)  # <-- add this line
                segments_json = json.loads(content)
                segments_raw = self.strip_code_fences(segments_json)  # <-- add this line
                segments = self.reconcile_boundaries(text, segments_raw)
                
                if isinstance(segments, list):
                    
                    # if self.debug:
                    #     print(f"Parsed segments: {segments}")
                    #     print(f"Verification: {''.join(segments)} == {text} -> {self.verify_segmentation(text, segments)}")
                    return segments

            except json.JSONDecodeError:
                if self.debug:
                    print(f"JSON parse failed: {content}")
            
            # Fallback: try to extract array from response
            array_match = re.search(r'\[(.*?)\]', content, re.DOTALL)
            if array_match:
                array_content = '[' + array_match.group(1) + ']'

                segments = json.loads(array_content)
                if isinstance(segments, list):
                    # if self.debug:
                    #     print(f"Fallback parsed segments: {segments}")
                    #     print(f"Verification: {''.join(segments)} == {text} -> {self.verify_segmentation(text, segments)}")
                    return segments
            
            raise ValueError(f"Could not parse LLM response: {content}")
            
        except Exception as e:
            if self.debug:
                print(f"Error in text segmentation mode: {e}")
                print(content)
            return [text]  # Return original text as single segment on failure
    
    # ❷  ────────────────────────────────────────────────────────────────────
    # NEW helper: prompt + call for the "≤ short_cutoff tokens" case
    def segment_by_indices(self,
                           text_span: str,
                           span_tokens: List[str],
                           temperature: float,
                           level: int) -> Optional[List[str]]:
        """
        Ask the LLM for integer *cut points* rather than sliced text when the
        span is short.  Returns the correctly reconstructed segments.
        """
        # Build a numbered token list to show the model
        enumerated = [f"{i}: '{tok}'" for i, tok in enumerate(span_tokens)]
        token_table = "\n".join(enumerated)

        # --- inside segment_by_indices() -------------------------------------
        max_segments = self.k               # e.g. 4
        max_cuts     = max_segments - 1     # cuts  = segments – 1

        system_prompt = (
            "You are a **phrase-level segmentation assistant**.\n"
            "Given a SHORT span (≤ {max_tok} tokens) and its token list with indices, "
            "return a JSON array of **cut-points** (0-based integers) where a new "
            "segment should START.\n\n"

            "Principles (in order of importance):\n"
            "1. **Meaningful units** – segments should be the largest contiguous group "
            "   of tokens that form a coherent phrase (e.g. determiner + noun, "
            "   adjective + noun, verb + particle).\n"
            "2. **Non-trivial cuts** – produce **at least one** cut-point but "
            f"no more than *{max_cuts}*, yielding up to *{max_segments}* segments in "
            "total; NEVER split into single-token segments unless a token truly stands "
            "alone (punctuation, conjunction, etc.).\n"
            "3. **Ascending order** – cut-points must be strictly increasing and "
            "between 1 and len(tokens)-1.\n"
            "4. **JSON only** – respond with the RAW JSON array (e.g. [3]) and nothing "
            "else.\n\n"

            "Guidelines:\n"
            "• Keep a leading article with its noun:  ['The', ' quick', ' brown', ' fox']\n"
            "  ⇒  cut-points [3]   →  ['The quick brown', ' fox']\n"
            "• Keep adjective(s) with the noun they modify.\n"
            "• Keep multi-word verbs together: [' jumps', ' over'] stays intact.\n"
            "• Attach sentence-final punctuation to the preceding word.\n"
            "• If the span is already one coherent phrase, output an empty array [].\n"
            "If you believe the span is already atomic, still output an empty array []."
            "Otherwise you MUST return ≥ 1 cut.\n"
            "If you start to output an empty array but then realise cuts are possible,"
            "**regenerate** your answer instead of sending the empty array.\n"
        ).format(max_tok=self.short_cutoff)
        # ----------------------------------------------------------------------


        user_prompt = (
            f"Text span:\n```{text_span}```\n\n"
            f"Tokens ({len(span_tokens)} total):\n{token_table}"
        )
        
        try: 
            raw = self._call_llm(system_prompt,
                     user_prompt,
                     temperature=temperature)

            # Robust, fence-free parse:
            cuts = literal_eval(re.search(r'\[.*\]', raw).group(0))

            if not cuts:                         # empty list ⇒ no split
                return None

            # Build segments from cut indices
            segments: List[str] = []
            start = 0
            for cut in cuts:
                segments.append(''.join(span_tokens[start:cut]))
                start = cut
            segments.append(''.join(span_tokens[start:]))

            return segments

        except Exception as e:
            # print(f"Error in token-index segmentation mode: {e}")
            return None                          # ← parse / regex / eval failed

    # ❸  ────────────────────────────────────────────────────────────────────

    def sentences_with_spans(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Sentence splitter for transcript-like text.

        • Leading spaces belong to the sentence **they precede**.
        • Trailing spaces after the final punctuation belong to the *next*
        sentence (or are dropped if none).
        That matches GPT-style tokenisers, which encode leading spaces as part
        of the following token (e.g. `' But'`).
        """
        # Regex:   ^\s*       → skip initial whitespace once
        #          .*?        → minimal grab until ...
        #          [.!?]      →   a sentence-ending punctuation
        #          (?=|\s)    →   *peek* at whitespace/end but don't consume it
        SENT_RE = re.compile(r'\s*.*?[.!?](?=\s|$)', re.DOTALL)

        out: List[Tuple[str, Tuple[int, int]]] = []
        idx = 0
        for m in SENT_RE.finditer(text):
            span_start, span_end = m.span()
            sent = text[span_start:span_end]
            out.append((sent, (span_start, span_end)))
            idx = span_end                        # next search resumes here

        # Handle a possible tail fragment without terminal punctuation
        if idx < len(text):
            out.append((text[idx:], (idx, len(text))))

        return out

    def segment_by_sentence(
            self,
            text: str,
            temperature: float = 0.1,
            level: int = 0,
        ) -> List[str]:
        """
        Long-span splitter that:
        1.  Numbers the sentences locally.
        2.  Asks GPT-4 for cut-points.
        3.  Re-assembles exact substrings.
        """
        sents = self.sentences_with_spans(text)
        N = len(sents)
        if N <= 1:
            return [text]

        numbered = "\n".join(f"[{i}] {sents[i][0]}" for i in range(N))

        content = self._call_llm(
            self.get_sentence_system_prompt(),
            numbered,
            temperature=temperature,
            max_tokens=200,
            model=self.model,
        )
        if self.debug:
            print("[segment_by_sentence]")
            print(f"LLM output: '{content}'")

        # --- parse cut-points -----
        try:
            cuts = json.loads(content)

            if (isinstance(cuts, list) and
                    all(isinstance(c, int) for c in cuts)):
                # enforce validity
                # ------------------------------------------------------------
                # 4. Build exact substrings from the original text
                # ------------------------------------------------------------
                cuts = sorted({c for c in cuts if 1 <= c < N})     # canonical, unique
                seg_bounds = [0] + cuts + [N]                      # sentinel at end
                segments = []

                for i in range(len(seg_bounds) - 1):
                    first_idx   = seg_bounds[i]                    # first sentence in seg
                    last_idx    = seg_bounds[i + 1] - 1            # last sentence in seg
                    start_char  = sents[first_idx][1][0]           # char span of first sent
                    end_char    = sents[last_idx][1][1]            # char span of last sent
                    segments.append(text[start_char:end_char])     # slice from original

                if self.verify_segmentation(text, segments):
                    return segments
        except Exception as e:
            print(f"Error in sentence-index segmentation mode: {e}")
            if self.debug:
                print("cut-point parse failed:", e)
                print(content)

        # fallback: return whole span
        return [text]


    def verify_segmentation(self, original: str, segments: List[str]) -> bool:
        """
        Verify that concatenating segments reproduces the original text.
        """
        concatenated = ''.join(segments)
        text_match = concatenated.rstrip() == original.rstrip()
        valid_count = len(segments) <= self.k  # Must have >1 segment unless single token

        # if self.debug:
        #     print(f"Verifying segmentation:")
        #     print(f"  Original: '{original}' (len={len(original)})")
        #     print(f"  Concatenated: '{concatenated}' (len={len(concatenated)})")
        #     print(f"  Original bytes: {original.encode('utf-8')}")
        #     print(f"  Concatenated bytes: {concatenated.encode('utf-8')}")
        return text_match and valid_count
    

    def find_token_ranges(
        self,
        segments: List[str],
        tokens: List[str],
        token_indices: List[int],
        start_token_idx: int = 0,
        text: Optional[str] = None,          # ← pass the *parent span* text
    ) -> List[Tuple[int, int]]:
        """
        Map each LLM-returned `segment` to a token slice [left, right).

        Strategy
        --------
        1. Locate the segment in `text` with str.find, starting at a running
        character cursor (so we pick the first non-overlapping match).
        2. Convert the character span [seg_start, seg_end) to token indices:
        • left  = first token whose *end* crosses seg_start
        • right = first token whose *start* is ≥ seg_end
        3. Advance both cursors and continue.

        Guarantees
        ----------
        • Every byte of the parent span is covered by exactly one child slice.
        • No token is skipped or duplicated, even when a boundary falls inside a
        token that begins with a leading space.
        """
        if text is None:
            raise ValueError("`text` (parent span) must be provided")

        ranges: List[Tuple[int, int]] = []
        char_cursor = token_indices[start_token_idx]   # absolute char offset
        tok_cursor  = start_token_idx                  # token index cursor

        for seg in segments:
            # ❶ find segment in text after the current char_cursor
            seg_start = text.find(seg, char_cursor)
            if seg_start == -1:
                # fallback: treat remaining text as one chunk
                seg_start = char_cursor
            seg_end = seg_start + len(seg)

            # ----- map left index: first token whose end > seg_start ----------
            left = tok_cursor
            while left < len(token_indices):
                t_end = token_indices[left] + len(tokens[left])
                if t_end > seg_start:
                    break
                left += 1

            # ----- map right index: first token whose start >= seg_end --------
            right = left
            while right < len(token_indices) and token_indices[right] < seg_end:
                right += 1

            ranges.append((left, right))

            # advance cursors for next segment
            char_cursor = seg_end
            tok_cursor  = right

        return ranges


    def _bump_total(self):
        with self._lock:
            self.total_spans += 1

    def _bump_fallback(self, tok_count: int, text: str, failed_segments: List[str]):
        with self._lock:
            self.fallback_spans       += 1
            self.fallback_token_total += tok_count
            self.fallback_lengths.append(tok_count)   # NEW – record the length
            if tok_count > 10:
                self.fallback_segments.append(failed_segments)
                self.fallback_texts.append(text)

    def _bump_leaf(self):
        """Thread-safe increment of the leaf counter."""
        with self._leaf_lock:
            self.leaf_count += 1
            # compute percentage
            pct = (self.leaf_count / self.total_tokens * 100) if self.total_tokens else 0

            if self.pbar is not None:
                # advance the bar by 1
                self.pbar.update(1)
            elif self.debug:
                # fallback to text print
                print(f"[Progress] leaf {self.leaf_count}/{self.total_tokens}")

            return self.leaf_count

    def _binary_split_tokens(self, span_tokens):
        """
        Deterministically split a list of tokens into TWO contiguous halves.

        • If the span has N tokens, the left segment gets ⌊N/2⌋ tokens,
        the right gets the rest.
        • All original spacing is preserved because tokens already hold
        their leading spaces.
        """
        mid = len(span_tokens) // 2            # floor division
        left  = ''.join(span_tokens[:mid])
        right = ''.join(span_tokens[mid:])
        return [left, right]

    def _print_stats(self) -> None:
        """
        Report how often the deterministic fallback was used and the
        mean token-length of those rescued spans.
        """
        if self.total_spans == 0:
            print("\nNo multi-token spans processed.")
            return

        fallback_rate = self.fallback_spans / self.total_spans
        avg_len = (self.fallback_token_total / self.fallback_spans
                if self.fallback_spans else 0)

        print(
            f"\nFallback used on {self.fallback_spans}/{self.total_spans} spans "
            f"({fallback_rate:.1%}); "
            f"average span length when fallback triggered: {avg_len:.2f} tokens"
        )
        

    def recursive_segment(
            self,
            text: str,
            tokens: List[str],
            token_indices: List[int],
            start_token_idx: int = 0,
            level: int = 0,
            max_temp: float = 1.0,
            end_token_idx: Optional[int] = None,      # ← NEW
    ) -> Dict[str, Any]:

        """
        Recursively segment text into hierarchical chunks.
        
        Args:
            text: Text to segment
            tokens: All tokens from original text
            token_indices: Character indices of all tokens
            start_token_idx: Starting token index
            level: Current recursion level
            max_temp: Maximum temperature to try
            
        Returns:
            Dictionary representing hierarchical segmentation
        """
        # if self.debug:
        #     print(f"\n--- RECURSIVE SEGMENT LEVEL {level} ---")
        #     print(f"Text: '{text}'")
        #     print(f"Start token idx: {start_token_idx}")
        

        if end_token_idx is not None:
            token_count = end_token_idx - start_token_idx
        else:
            # Calculate how many tokens this text segment contains
            # by reconstructing from original tokens
            reconstructed = ""
            token_count = 0
            temp_idx = start_token_idx
            
            while temp_idx < len(tokens) and len(reconstructed) < len(text):
                if reconstructed + tokens[temp_idx] == text[:len(reconstructed + tokens[temp_idx])]:
                    reconstructed += tokens[temp_idx]
                    token_count += 1
                    temp_idx += 1
                else:
                    break
        #     if self.debug:
        #         print(f"Reconstructed: '{reconstructed}'")
        #         print(f"Match: {reconstructed == text}")
        
        # if self.debug:
        #     print(f"Token count for this segment: {token_count}")


        # --- bookkeeping for fallback statistics --------------------------
        if token_count > 1:                  # only spans that *can* be split
            self.total_spans += 1

        
        # If this is a single token, return as leaf
        if token_count <= 1:
            leaf_num = self._bump_leaf()
            if self.debug:
                print(f"[Leaf {leaf_num}] level={level}, tokens {start_token_idx}:{start_token_idx+token_count} → “{text!r}”")

            return {
                'text': text,
                'token_range': (start_token_idx, start_token_idx + token_count),
                'level': level,
                'children': None
            }
        
        # Decide which segmentation strategy to use
        use_index_mode = (1 < token_count <= self.short_cutoff)
        use_sentence_mode = (token_count >= 512)


        temperature = 0.1
        segments = None
        successful_segmentation = False

        while temperature <= max_temp:
            if use_index_mode:
                # --- New path for short spans ----------------------------
                span_tokens = tokens[start_token_idx : start_token_idx + token_count]
                segments = self.segment_by_indices(text, span_tokens,
                                                   temperature, level)
            elif use_sentence_mode:    
                segments = self.segment_by_sentence(text, temperature, level)

            else:
                # --- Original path --------------------------------------
                segments = self.segment_text(text, temperature, level)

            # Verify & accept
            if (segments and len(segments) > 1 and
                    self.verify_segmentation(text, segments) and
                    all(seg.strip() for seg in segments)):
                successful_segmentation = True
                break

            temperature += 0.2

        if not successful_segmentation and token_count > 1:
            failed_segments = segments.copy() if segments else []
            # --- deterministic binary split fallback --------------------------
            span_tokens = tokens[start_token_idx : start_token_idx + token_count]

            segments = self._binary_split_tokens(span_tokens)

            # accept the fallback only if it reconstructs perfectly
            if self.verify_segmentation(text, segments):
                successful_segmentation = True
                self._bump_fallback(token_count, text, failed_segments)   # ← single, thread-safe call

        # ----------------------------------------------------------------------

        # If segmentation failed, return as leaf node
        if not successful_segmentation or len(segments) <= 1:
            leaf_num = self._bump_leaf()
            if self.debug:
                print(f"[Leaf {leaf_num}] (fallback) level={level}, tokens {start_token_idx}:{start_token_idx+token_count} → “{text!r}”")
            return {
                'text': text,
                'token_range': (start_token_idx, start_token_idx + token_count),
                'level': level,
                'children': None  
            }
        
        # Find token ranges for segments
        token_ranges = self.find_token_ranges(
            segments,
            tokens,
            token_indices,
            start_token_idx,
            text=text          # ← parent span string
        )

        # if self.debug:
        #     print(f"Finding token ranges for segments: {segments}")
        #     print(f"Token ranges: {token_ranges}")

        # Build canonical text for every slice so leading / trailing spaces match
        canonical_segments = [
            ''.join(tokens[l:r]) for (l, r) in token_ranges
        ]

        # ──────────────────────────────────────────────────────────────────────
        # Recursively segment each chunk
        # ──────────────────────────────────────────────────────────────────────
        children = [None] * len(segments)

        if self.multithread and len(segments) > 1:
            # depth-local thread pool
            with cf.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                future_to_idx = {
                    pool.submit(
                        self.recursive_segment,
                        canonical_segments[idx],           # ← use canonical text
                        tokens,
                        token_indices,
                        tr[0],                 # start idx
                        level + 1,
                        max_temp,
                        end_token_idx=tr[1]    # ← pass the right boundary
                    ): idx
                    for idx, (seg, tr) in enumerate(zip(segments, token_ranges))
                }
                for fut in cf.as_completed(future_to_idx):
                    children[future_to_idx[fut]] = fut.result()
        else:
            # sequential fallback
            for idx, (seg, tr) in enumerate(zip(segments, token_ranges)):
                children[idx] = self.recursive_segment(
                    canonical_segments[idx],           # ← here too
                    tokens,
                    token_indices,
                    tr[0],
                    level + 1,
                    max_temp,
                    end_token_idx=tr[1]       # ← same here
                )

        return {
            'text': text,
            'token_range': (start_token_idx, start_token_idx + token_count),
            'level': level,
            'children': children
        }


    def chunk_text(self, text: str) -> Dict[str, Any]:
        """
        Main method to perform hierarchical chunking.
        
        Args:
            text: Input text to chunk
            
        Returns:
            Dictionary with hierarchical segmentation and metadata
        """
        # Step 1: Tokenize the entire input
        tokens, token_indices = self.tokenize_text(text)
        # set total tokens so progress has a denominator
        self.total_tokens = len(tokens)

        # Initialize tqdm bar once we know how many leaves to expect
        # (we set total to number of tokens, since each token becomes exactly one leaf)
        self.pbar = tqdm(
            total=len(tokens),
            desc="Hierarchical chunking",
            unit="leaf",
            ncols=80,
        )        

        if self.debug:
            print(f"[chunk_text] → total tokens = {len(tokens)}")
        
        # Step 2: Recursive segmentation
        hierarchy = self.recursive_segment(text, tokens, token_indices)

        # Done — close the bar
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

        self._print_stats()        # ← optional helper that prints rate & avg length
        return {
            'original_text': text,
            'tokens': tokens,
            'token_indices': token_indices,
            'hierarchy': hierarchy
        }

    
    def print_hierarchy(self, node: Dict[str, Any], indent: int = 0):
        """Print the hierarchical structure in a readable format."""
        prefix = "  " * indent
        token_range = node['token_range']
        print(f"{prefix}Level {node['level']}: '{node['text']}' (tokens {token_range[0]}-{token_range[1]})")
        
        if node['children']:
            for child in node['children']:
                self.print_hierarchy(child, indent + 1)


#========================================================================================================
#========================================================================================================
#========================================================================================================

"""
Helper functions for processing text
"""


def hierarchy_to_span_tree(node: dict) -> dict:
    """
    Convert the chunker’s node format

        {'text': str,
         'token_range': (start_idx, end_idx),
         'level': int,
         'children': [ ... ] or None}

    into the minimal span dictionary used by downstream code:

        {'start': int,
         'end': int,
         'children': [ ... ]}

    Notes
    -----
    • `start` is inclusive, `end` is exclusive (Python-slice style).  
    • If the node has no children, an *empty list* is stored (not None) so the
      output tree is homogenous and easy to traverse.
    """
    start, end = node["token_range"]
    children_nodes = node.get("children") or []

    return {
        "start": start,
        "end":   end,
        "children": [hierarchy_to_span_tree(child) for child in children_nodes]
    }

def bracketize_hierarchy(node: dict, level: int = 0) -> str:
    """
    Recursively wrap each child segment in square brackets, so that
    the final string visualises *every* split point.

    Example for top-level ["The quick brown", " fox"]:
        '[The quick brown][ fox]'

    Nested splits generate nested brackets:
        '[[The][ quick brown]][ fox]'

    Parameters
    ----------
    node   : a node from `chunker.chunk_text(... )["hierarchy"]`
    level  : recursion depth (the root is 0).  Kept only in case you want
             different bracket symbols for different levels.

    Returns
    -------
    A single string equal in bytes to the original text except for the
    added '[' and ']' markers.
    """
    # Leaf → just return raw text
    if not node.get("children"):
        return node["text"]

    # Recursively bracket each child, then glue them together
    parts = [bracketize_hierarchy(child, level + 1) for child in node["children"]]

    # Surround every child with its own pair of brackets
    bracketed_children = "".join(f"[{p}]" for p in parts)

    return bracketed_children


def clean_input_text(text: str) -> str:
    """
    Sanitize user text before chunking.

    Steps
    -----
    0.  Remove an outer ```fence``` if present.
    1.  Normalise CR/LF line-break variants.
    2.  Replace newline / tab with a single space.
    3.  *Insert a space* after ".", "!" or "?" if not already followed by
        whitespace and the next char starts a new sentence (capital or digit).
    4.  Collapse runs of   2+ spaces  into a single space.
    5.  Trim leading / trailing whitespace.
    6.  Drop control characters (except the newline we already turned to space).

    The resulting string preserves interior spaces and punctuation, ensuring
    the chunker & verifier see the exact same byte sequence.
    """
    # ── 0️⃣ remove outer triple-backtick fence ───────────────────────────
    fence = re.compile(r"^\s*```[\w+-]*\n?(.*?)\n?```\s*$", re.DOTALL)
    m = fence.match(text)
    if m:
        text = m.group(1)

    # ── 1️⃣ universal newlines ──────────────────────────────────────────
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # ── 2️⃣ newline / tab → single space ────────────────────────────────
    text = re.sub(r'[\n\t]+', ' ', text)

    # ── 3️⃣ fix missing sentence space  (".They" → ". They") ────────────
    sent_end = re.compile(r'([.!?])([A-Z0-9])')
    text = sent_end.sub(r'\1 \2', text)

    # ── 4️⃣ collapse multiple spaces ────────────────────────────────────
    text = re.sub(r' {2,}', ' ', text)

    # ── 5️⃣ trim outer spaces ───────────────────────────────────────────
    text = text.strip()

    # ── 6️⃣ strip other control chars ───────────────────────────────────
    text = ''.join(ch for ch in text
                   if unicodedata.category(ch)[0] != 'C' or ch == '\n')
    
    # # ── 7️⃣ Quote normalisation & escape cleanup ───────────────────────
    # _DBL_QUOTES = {
    #     '"',               # U+0022  straight
    #     '“', '”',          # U+201C, U+201D  left/right curly
    #     '„', '‟',          # U+201E, U+201F  low/high double comma
    #     '❝', '❞',          # U+275D, U+275E  heavy quotation marks
    #     '＂',              # U+FF02  full-width
    # }
    # _QUOTE_TRANS = str.maketrans({ch: "'" for ch in _DBL_QUOTES})
    # _ESCAPE_RE  = re.compile(r"""\\(['"])""")   # matches \" or \'

    # text = text.translate(_QUOTE_TRANS)      # fancy → '
    # text = _ESCAPE_RE.sub(r"\1", text)       # \"  or  \'  → " / '

    return text

#========================================================================================================
#========================================================================================================
#========================================================================================================

"""
Tree class for hierarchical spans
"""

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib
import numpy as np
import tiktoken
from typing import List, Optional, Union, Any, Tuple, Iterator, Dict

# Use a math font
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 10


def range_tree_to_counts(node: Dict[str, Any]) -> Union[int, List]:
    """
    Given a tree of {"start":i, "end":j, "children":[...]} returns
    either an int (for leaves) or a list of ints/lists for internal nodes.
    """
    children = node.get("children", [])
    if not children:
        # leaf: exactly the number of tokens in this span
        return node["end"] - node["start"]
    # internal: convert each child
    return [range_tree_to_counts(child) for child in children]



class Node:
    def __init__(self, count: int, children: Optional[List['Node']] = None, token: Optional[str] = None):
        self.value = count
        self.children = children or []
        self.token = token


def build_token_tree(partition: Union[int, List[Any]], token_iter: Iterator[int], encoding) -> Node:
    """
    Recursively builds a tree of Nodes from a partition and a token iterator.
    Groups `count` token IDs, decodes them together to preserve full text.
    """
    if isinstance(partition, int):
        count = partition
        # Collect token IDs
        ids = [next(token_iter, None) for _ in range(count)]
        ids = [tid for tid in ids if tid is not None]
        text = encoding.decode(ids)
        return Node(count, token=text)
    else:
        children = [build_token_tree(p, token_iter, encoding) for p in partition]
        total = sum(child.value for child in children)
        return Node(total, children=children)


def traverse_tree(node: Node) -> Tuple[List[Tuple[int, Dict[str, Any]]],
                                       List[Tuple[int, int]]]:
    nodes: List[Tuple[int, Dict[str, Any]]] = []
    edges: List[Tuple[int, int]] = []

    def _traverse(n: Optional[Node], parent: Optional[Node] = None):
        if n is None:
            return

        node_id = id(n)

        if n.children:                                 # ── internal node
            nodes.append((node_id, {'value': n.value}))
            if parent:
                edges.append((id(parent), node_id))
            for child in n.children:
                _traverse(child, n)

        else:                                          # ── leaf node
            if n.token is None:
                return

            # ── keep pure whitespace; otherwise trim ───────────────────
            if n.token.strip() == "":
                label = n.token          # exactly as given (may look blank)
            else:
                label = n.token.strip()  # drop leading/trailing spaces

            nodes.append((node_id, {'value': label}))
            if parent:
                edges.append((id(parent), node_id))

    _traverse(node)
    return nodes, edges



def get_positions(node: Node, y: float = 0, positions: Optional[Dict[int, Tuple[float, float]]] = None,
                  leaf_x: Optional[List[int]] = None) -> Dict[int, Tuple[float, float]]:
    if positions is None:
        positions = {}
    if leaf_x is None:
        leaf_x = [0]
    if node.children:
        for child in node.children:
            get_positions(child, y - 1, positions, leaf_x)
        xs = [positions[id(c)][0] for c in node.children if id(c) in positions]
        positions[id(node)] = (sum(xs) / len(xs), y)
    else:
        positions[id(node)] = (leaf_x[0], y)
        leaf_x[0] += 1
    return positions

def plot_tree_vertical(nodes: List[Tuple[int, Dict[str, Any]]],
                       edges: List[Tuple[int, int]],
                       pos: Dict[int, Tuple[float, float]],
                       figsize: Tuple[float, float] = (8, 12),
                       show_text: bool = True,
                       out_file: str = "token_tree_vertical.png",
                       showfig=True,            
                    ):
    """
    Plot the token tree **top-to-bottom** instead of left-to-right.

    Rotation trick:
        new_x = old_y
        new_y = -old_x        # minus sign keeps root at the top
    """
    # ── rotate the layout ───────────────────────────────────────────────
    # vpos = {nid: (-y, -x) for nid, (x, y) in pos.items()}
    vpos = {nid: (y, -x) for nid, (x, y) in pos.items()}


    # ── build graph ─────────────────────────────────────────────────────
    plt.figure(figsize=figsize)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    internal = [nid for nid, _ in nodes if G.out_degree(nid) > 0]
    leaf     = [nid for nid, _ in nodes if G.out_degree(nid) == 0]

    my_blue  = '#42c5f5'
    my_green = '#42f5b3'
    color_map = {nid: (my_blue if nid in leaf else my_green) for nid, _ in nodes}

    # nodes & edges
    nx.draw_networkx_nodes(G, vpos, nodelist=internal,
                           node_size=800,
                           node_color=[color_map[n] for n in internal],
                           edgecolors='black')
    nx.draw_networkx_nodes(G, vpos, nodelist=leaf,
                           node_size=1200,
                           node_color=[color_map[n] for n in leaf],
                           edgecolors='black')
    nx.draw_networkx_edges(G, vpos, arrows=False)

    # labels
    labels_int = {nid: data['value'] for nid, data in nodes if nid in internal}
    nx.draw_networkx_labels(G, vpos, labels_int, font_size=20,
                            font_family='STIXGeneral')

    if show_text:
        labels_leaf = {nid: data['value']
                       for nid, data in nodes if nid in leaf and data['value']}
        nx.draw_networkx_labels(G, vpos, labels_leaf, font_size=14,
                                font_family='STIXGeneral')

    # tight limits
    xs = [x for x, _ in vpos.values()]
    ys = [y for _, y in vpos.values()]
    plt.xlim(min(xs) - 0.5, max(xs) + 0.5)
    plt.ylim(min(ys) - 0.5, max(ys) + 0.5)

    plt.axis('off')
    # plt.savefig(out_file, bbox_inches='tight', dpi=100)

    if showfig:
        plt.show()
    else:
        plt.close()


def plot_chunks_with_text_vertical(
    partition: Union[int, List[Any]],
    text: str,
    *,
    figsize: Tuple[float, float] = (30, 4),
    show_text: bool = True,
    showfig: bool = True,
    out_file: Optional[str] = None
) -> Node:
    
    encoding   = tiktoken.get_encoding("cl100k_base")
    token_ids  = encoding.encode(text)
    token_iter = iter(token_ids)

    root = build_token_tree(partition, token_iter, encoding)
    nodes, edges = traverse_tree(root)
    pos          = get_positions(root)

    plot_tree_vertical(
        nodes,
        edges,
        pos,
        figsize=figsize,
        show_text=show_text,
        out_file=out_file or "token_tree_vertical.png",
        showfig=showfig
    )

    return root


#========================================================================================================
#========================================================================================================
#========================================================================================================

"""
High level chunker functions.
"""

from typing import Optional, List, Tuple, Dict, Any

def chunk_main(
    input_text: str,
    *,
    model: str = "gpt-4o",
    client: Optional[object] = None,     # e.g. Together(...)
    api_key: Optional[str] = None,       # ignored if client is given
    multithread: bool = False,
    debug: bool = False,
    k: int = 4,
) -> Tuple[Dict[str, Any], List[int]]:    # ← use typing.List / Tuple
    """
    Run the hierarchical chunker on `input_text`.

    Parameters
    ----------
    model        : model name (ignored if `client` hard-codes its own default)
    client       : an instantiated client with `.chat.completions.create`
    api_key      : only used to build the default OpenAI client
    multithread  : enable nested thread-pool segmentation
    k            : max segments per LLM call

    Returns
    -------
    (result, lengths)
        result   : full dict returned by `chunk_text`
        lengths  : list[int] of span lengths where deterministic fallback fired
    """

    chunker = HierarchicalChunker(
        api_key     = api_key,
        model       = model,
        client      = client,
        k           = k,
        multithread = multithread,
        debug       = debug,
    )

    result  = chunker.chunk_text(input_text)
    lengths = chunker.fallback_lengths

    print(f"Fallback lengths: {lengths}")
    return result, lengths, chunker


def main(
    input_text: str,
    *,
    model: str = "gpt-4o",
    client: Optional[object] = None,
    api_key: Optional[str] = None,
    multithread: bool = False,
    debug: bool = False,
    k: int = 4,
    figsize: Tuple[int, int] = (10, 40),
    show_text: bool = True,
    showfig: bool = True,
) -> Dict[str, Any]:
    """
    Clean text, chunk it, plot the vertical tree, and return a dict with:
      - 'original_text': the cleaned input
      - 'tokens': token list
      - 'token_indices': character indices for each token
      - 'hierarchy': raw hierarchy tree from the chunker
      - 'fallback_lengths': list of fallback span-lengths
      - 'span_tree': the minimal span-tree dict
      - 'partition': token-count partition used for plotting
    """
    # 1. clean + chunk
    cleaned = clean_input_text(input_text)
    chunk_result, fallback_lengths, chunker = chunk_main(
        cleaned,
        model       = model,
        client      = client,
        api_key     = api_key,
        multithread = multithread,
        debug       = debug,
        k           = k,
    )

    # 2. build the span-tree and partition
    span_tree  = hierarchy_to_span_tree(chunk_result["hierarchy"])
    partition  = range_tree_to_counts(span_tree)
    fallback_segments = chunker.fallback_segments
    fallback_texts = chunker.fallback_texts

    root = plot_chunks_with_text_vertical(partition, cleaned,
                                   figsize=figsize,
                                   show_text=show_text,
                                   showfig=showfig)

    # 4. return everything in a flat dict
    return {
        "original_text":    cleaned,
        "tokens":           chunk_result["tokens"],
        "token_indices":    chunk_result["token_indices"],
        "hierarchy":        chunk_result["hierarchy"],
        "fallback_lengths": fallback_lengths,
        "fallback_segments": fallback_segments,  # ← list of segments where fallback was used
        "fallback_texts":   fallback_texts,     # ← list of texts where fallback was used
        "span_tree":        span_tree,
        "partition":        partition,
    }




#========================================================================================================
#========================================================================================================
#========================================================================================================

"""
Tree statistics
"""

from collections import deque, defaultdict
from typing import List, Any
import os

def nodes_per_level(root: Any) -> List[int]:
    """
    Traverse the node tree and count how many nodes appear at each depth level.
    Returns a list where index `i` is the number of nodes at level `i`.
    """
    counts: defaultdict[int, int] = defaultdict(int)
    queue = deque([(root, 0)])
    while queue:
        node, level = queue.popleft()
        counts[level] += 1
        for child in getattr(node, 'children', []):
            queue.append((child, level + 1))
    max_level = max(counts.keys(), default=-1)
    return [counts[i] for i in range(max_level + 1)]


def branching_ratios_per_level(root: Any) -> List[List[int]]:
    """
    Traverse the node tree and collect each node's number of children, grouped by depth level.
    Returns a list of lists: element `i` is a list of branching counts at level `i`.
    """
    ratios: defaultdict[int, List[int]] = defaultdict(list)
    queue = deque([(root, 0)])
    while queue:
        node, level = queue.popleft()
        child_count = len(getattr(node, 'children', []))
        ratios[level].append(child_count)
        for child in getattr(node, 'children', []):
            queue.append((child, level + 1))
    max_level = max(ratios.keys(), default=-1)
    return [ratios[i] for i in range(max_level + 1)]


def chunk_sizes_per_level(root: Any) -> List[List[int]]:
    """
    Traverse the node tree and collect each node's `value`, grouped by depth level.
    Returns a list of lists: element `i` is a list of node values at level `i`.
    """
    values: defaultdict[int, List[int]] = defaultdict(list)
    queue = deque([(root, 0)])          # (node, depth)
    
    while queue:
        node, level = queue.popleft()
        
        # Record this node's value at its depth level
        values[level].append(getattr(node, 'value', None))
        
        # Enqueue children (if any) one level deeper
        for child in getattr(node, 'children', []):
            queue.append((child, level + 1))
    
    # Convert sparse defaultdict → dense list with contiguous levels [0 … max_level]
    max_level = max(values.keys(), default=-1)
    return [values[i] for i in range(max_level + 1)]

def dist_per_level(chunk_sizes: List[List[int]]) -> List[List[int]]:
    dist_per_level = []
    N = chunk_sizes[0][0]
    levels = len(chunk_sizes)
    for l in range(levels):
        dist, bin_edges = np.histogram(chunk_sizes[l], bins=np.arange(1,N+1,1), density=True)
        dist_per_level.append(dist)

    return dist_per_level, bin_edges

def tree_analysis(
    results_dir: str,
    model: str,
    # *,
    # encoding_name: str = "cl100k_base"
) -> Dict[str, Dict[str, Any]]:
    """
    For every `*_result.json` in `results_dir`, rebuild the Node tree,
    compute its statistics, and return an *ordered* dict mapping:

        "<story_name>" → {
            "root": <Node>,
            "num_tokens": int,
            "chunk_sizes": List[List[int]],
            "distribution": {"dist": List[int], "bin_edges": List[float]},
            "nodes_per_level": List[int],
            "branching_ratios": List[List[int]],
            "partition": <original partition data>
        }

    The dict is ordered by ascending `num_tokens`.
    """
    # prepare tokenizer
    encoding = tiktoken.get_encoding("cl100k_base")

    # first collect everything unsorted
    all_stats: Dict[str, Dict[str, Any]] = {}

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith("_result.json"):
            continue

        story_name = fname[:-len("_result.json")]
        path = os.path.join(results_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)

        text      = result["original_text"]
        partition = result["partition"]
        tokens    = result["tokens"]

        # rebuild the tree
        token_ids   = encoding.encode(text)
        root        = build_token_tree(partition, iter(token_ids), encoding)

        # compute stats
        chunk_sizes  = chunk_sizes_per_level(root)
        dist, bins   = dist_per_level(chunk_sizes)
        Cs           = nodes_per_level(root)
        branches     = branching_ratios_per_level(root)

        all_stats[story_name] = {
            "num_tokens": len(tokens),
            "chunk_sizes": chunk_sizes,
            "distribution": {
                "dist": dist,
                "bin_edges": bins
            },
            "nodes_per_level": Cs,
            "branching_ratios": branches,
            "partition": partition
        }

    # now sort by num_tokens and build a new dict in that order
    sorted_stats = dict(
        sorted(
            all_stats.items(),
            key=lambda item: item[1]["num_tokens"]
        )
    )

    return sorted_stats

def collect_level_data(
    stats: Dict[str, Dict[str, Any]],
    level: int
) -> Tuple[List[Any], List[Any], List[int]]:
    """
    For each story in `stats`, if it has a distribution and bin_edges at `level`,
    collect:
      - dist[level]
      - bin_edges[level]
      - num_tokens

    Returns three parallel lists: (dists, bin_edges_list, Ns)
    """
    dists: List[Any]      = []
    bin_edges_list: List[Any] = []
    Ns: List[int]         = []
    s_samples: List[List[float]] = []


    for story_name, info in stats.items():
        dist_arr     = info["distribution"]["dist"]
        bin_edges_arr= info["distribution"]["bin_edges"]
        N            = info["num_tokens"]
        chunk_sizes   = info["chunk_sizes"]            # raw n values


        # only include if this level exists in both lists
        if level < len(dist_arr) and level < len(bin_edges_arr):
            dists.append(dist_arr[level-1]) # level-1 for 0-indexing
            bin_edges_list.append(bin_edges_arr)
            Ns.append(N)

            # --- rescale n → s = n/N ---
            s_vals = [n / N for n in chunk_sizes[level-1]]
            s_samples.append(s_vals)

    return dists, bin_edges_list, Ns, s_samples