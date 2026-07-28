import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
from typing import List, Dict, Tuple, Callable

class PCFGGenerator:
    """Probabilistic Context-Free Grammar generator.
    
    Each non-terminal has two production rules with equal probability (0.5 each).
    The grammar generates sequences of terminals {a, b, c}.

    Correlation between count(a) and count(b) is no longer handled inside this
    class.  Instead, use ``build_pools`` to pre-generate correlated / uncorrelated
    string pools and mix them at the desired ratio during training.
    """
    
    def __init__(self):
        # Define production rules: each non-terminal maps to 2 possible expansions
        # Format: symbol -> [(prob, [expansion]), (prob, [expansion])]
        self.rules = {
            's': [(0.5, ['r', 'q']), (0.5, ['q', 'p'])],
            'p': [(0.5, ['m', 'n', 'o']), (0.5, ['n', 'o', 'm'])],
            'q': [(0.5, ['n', 'm', 'o']), (0.5, ['m', 'n'])],
            'r': [(0.5, ['o', 'm']), (0.5, ['m', 'o', 'n'])],
            'm': [(0.5, ['l', 'j']), (0.5, ['j', 'l', 'k'])],
            'n': [(0.5, ['k', 'j', 'l']), (0.5, ['l', 'j', 'k'])],
            'o': [(0.5, ['l', 'k', 'j']), (0.5, ['k', 'j'])],
            'j': [(0.5, ['h', 'i']), (0.5, ['i', 'h'])],
            'k': [(0.5, ['h', 'g', 'i']), (0.5, ['g', 'h', 'i'])],
            'l': [(0.5, ['i', 'h', 'g']), (0.5, ['h', 'i', 'g'])],
            'g': [(0.5, ['d', 'f', 'e']), (0.5, ['f', 'e', 'd'])],
            'h': [(0.5, ['e', 'd', 'f']), (0.5, ['d', 'e', 'f'])],
            'i': [(0.5, ['e', 'f', 'd']), (0.5, ['f', 'd', 'e'])],
            'd': [(0.5, ['c', 'a']), (0.5, ['a', 'b', 'c'])],
            'e': [(0.5, ['c', 'b']), (0.5, ['c', 'a', 'b'])],
            'f': [(0.5, ['c', 'b', 'a']), (0.5, ['b', 'a'])],
        }
        
        # Terminals (base symbols that don't expand further)
        self.terminals = {'a', 'b', 'c'}
    
    def _expand_symbol(self, symbol: str) -> List[str]:
        """Expand a single symbol according to production rules."""
        if symbol in self.terminals:
            return [symbol]
        
        if symbol not in self.rules:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        # Choose one of the two production rules with equal probability
        probs, expansions = zip(*self.rules[symbol])
        chosen_expansion = random.choices(expansions, weights=probs)[0]
        
        return chosen_expansion
    
    def generate(self, start_symbol: str = 's', max_iterations: int = 1000) -> str:
        """Generate a complete string from the grammar.
        
        Args:
            start_symbol: Starting non-terminal (default 's')
            max_iterations: Maximum expansion iterations to prevent infinite loops
        
        Returns:
            Generated string of terminals
        """
        symbols = [start_symbol]
        iterations = 0
        
        while iterations < max_iterations:
            # Check if all symbols are terminals
            if all(s in self.terminals for s in symbols):
                break
            
            # Find first non-terminal and expand it
            new_symbols = []
            for symbol in symbols:
                if symbol in self.terminals:
                    new_symbols.append(symbol)
                else:
                    # Expand this non-terminal
                    expansion = self._expand_symbol(symbol)
                    new_symbols.extend(expansion)
            
            symbols = new_symbols
            iterations += 1
        
        return ''.join(symbols)
    
    def generate_chunk(self, chunk_size: int = 250) -> str:
        """Generate a string and subsample a chunk of the given size.

        Args:
            chunk_size: Number of characters to return.

        Returns:
            Subsampled chunk of the generated string.
        """
        full_string = self.generate()
        while len(full_string) < chunk_size:
            full_string += self.generate()
        if len(full_string) == chunk_size:
            return full_string
        start_idx = random.randint(0, len(full_string) - chunk_size)
        return full_string[start_idx:start_idx + chunk_size]



# ---------------------------------------------------------------------------
# Pool builder
# ---------------------------------------------------------------------------

def build_pools(
    pcfg_gen: PCFGGenerator,
    n_correlated: int,
    n_uncorrelated: int,
    chunk_size: int = 250,
    window: int = 40,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """Pre-build correlated and uncorrelated PCFG string pools.

    **Correlated** strings satisfy ``count('a') == count('b') + 1`` in the
    last ``window`` characters (i.e. the same region the count tasks operate
    on).  **Uncorrelated** strings are naturally generated with no filter.

    Both pools are filled in a single pass:
    - Every generated string is added to the uncorrelated pool until it is full.
    - Strings whose last ``window`` chars have count(a) == count(b) + 1 go
      into the correlated pool.

    Args:
        pcfg_gen: PCFGGenerator instance.
        n_correlated: Number of correlated strings to collect.
        n_uncorrelated: Number of uncorrelated (natural) strings to collect.
        chunk_size: Length of each PCFG chunk.
        window: Number of trailing characters to check the correlation in
            (should match the count-task window, default 40).
        verbose: If True, print progress updates.

    Returns:
        ``{'correlated': [...], 'uncorrelated': [...]}``
    """
    correlated: List[str] = []
    uncorrelated: List[str] = []
    total_generated = 0

    if verbose:
        print(
            f"Building PCFG pools: {n_correlated:,} correlated + "
            f"{n_uncorrelated:,} uncorrelated strings (window={window}) …"
        )

    while len(correlated) < n_correlated:
        chunk = pcfg_gen.generate_chunk(chunk_size)
        total_generated += 1

        if len(uncorrelated) < n_uncorrelated:
            uncorrelated.append(chunk)

        tail = chunk[-window:]
        if tail.count('a') == tail.count('b') + 1:
            correlated.append(chunk)

        if verbose and total_generated % 50_000 == 0:
            rate = len(correlated) / total_generated
            print(
                f"  generated {total_generated:,} | correlated {len(correlated):,}/{n_correlated:,} "
                f"({100 * rate:.1f}% acceptance)"
            )

    # Top up uncorrelated pool if not yet full (rare edge-case)
    while len(uncorrelated) < n_uncorrelated:
        uncorrelated.append(pcfg_gen.generate_chunk(chunk_size))

    if verbose:
        rate = len(correlated) / max(total_generated, 1)
        print(
            f"Pools ready — {len(correlated):,} correlated, {len(uncorrelated):,} uncorrelated "
            f"from {total_generated:,} total generations ({100 * rate:.1f}% acceptance rate)."
        )

    return {'correlated': correlated, 'uncorrelated': uncorrelated}


def collate_fn(batch, tokenizer):
    """Collate function to pad sequences in a batch."""
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids = []
    target_ids = []
    answer_positions = []
    
    for item in batch:
        input_id = item['input_ids']
        target_id = item['target_ids']
        
        pad_len = max_len - len(input_id)
        input_ids.append(F.pad(input_id, (0, pad_len), value=tokenizer.pad_id))
        # Always pad targets with -100 so padding is ignored by loss
        target_ids.append(F.pad(target_id, (0, pad_len), value=-100))
        answer_positions.append(item['answer_positions'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'target_ids': torch.stack(target_ids),
        'answer_positions': answer_positions,
    }


class TaskRegistry:
    """Registry for PCFG tasks.
    
    Each task takes a PCFG string and returns (task_definition, answer).
    Tasks are registered with a name and can be easily added or removed.
    """
    
    def __init__(self):
        self.tasks: Dict[str, Callable[[str], Tuple[str, str]]] = {}
    
    def register(self, name: str, task_fn: Callable[[str], Tuple[str, str]]):
        """Register a new task.
        
        Args:
            name: Task name (identifier)
            task_fn: Function that takes PCFG string and returns (task_def, answer)
        """
        self.tasks[name] = task_fn
    
    def get_task(self, name: str) -> Callable[[str], Tuple[str, str]]:
        """Get a task function by name."""
        if name not in self.tasks:
            raise ValueError(f"Task '{name}' not found. Available: {list(self.tasks.keys())}")
        return self.tasks[name]
    
    def apply_task(self, name: str, pcfg_string: str) -> Tuple[str, str]:
        """Apply a task to a PCFG string.
        
        Returns:
            (task_definition, answer)
        """
        task_fn = self.get_task(name)
        return task_fn(pcfg_string)

# Task 1 (C): Count occurrences of a token in last N positions
def count_char_task(pcfg_string: str, char: str = 'a', window: int = 40) -> Tuple[str, str]:
    """Count occurrences of a character in the last N positions."""
    last_n = pcfg_string[-window:]
    count = last_n.count(char)
    task_def = f"C{char}{window}"
    answer = str(count)
    return task_def, answer

# Task 2 (CC): Count occurrences of substring in last N positions
def count_composition_task(pcfg_string: str, substring: str = 'aa', window: int = 40) -> Tuple[str, str]:
    """Count occurrences of substring in the last N positions."""
    last_n = pcfg_string[-window:]
    # Count overlapping occurrences
    count = 0
    for i in range(len(last_n) - len(substring) + 1):
        if last_n[i:i+len(substring)] == substring:
            count += 1
    task_def = f"CC{substring}{window}"
    answer = str(count)
    return task_def, answer

# Task 3 (I): Index from EOT when char occurred for the Nth time
def index_occurrence_task(pcfg_string: str, char: str = 'a', occurrence: int = 6) -> Tuple[str, str]:
    """Find index from EOT token when char occurred for the Nth time."""
    # Count from the end backwards
    count = 0
    for i in range(len(pcfg_string) - 1, -1, -1):
        if pcfg_string[i] == char:
            count += 1
            if count == occurrence:
                # Index from the end (EOT is at position 0 when counting backwards)
                index_from_end = len(pcfg_string) - 1 - i
                task_def = f"I{char}{occurrence}"
                answer = str(index_from_end)
                return task_def, answer
    
    # If char didn't occur N times, return -1
    task_def = f"I{char}{occurrence}"
    answer = "-1"
    return task_def, answer

# Task 4 (IC): Index from EOT when substring occurred for the Nth time
def index_composition_task(pcfg_string: str, substring: str = 'aa', occurrence: int = 6) -> Tuple[str, str]:
    """Find index from EOT when substring occurred for the Nth time."""
    # Find all occurrences (overlapping)
    positions = []
    for i in range(len(pcfg_string) - len(substring) + 1):
        if pcfg_string[i:i+len(substring)] == substring:
            positions.append(i)
    
    # Count from the end
    if len(positions) >= occurrence:
        # Get the Nth occurrence from the end
        pos = positions[-(occurrence)]
        # Index from the end
        index_from_end = len(pcfg_string) - 1 - pos
        task_def = f"IC{substring}{occurrence}"
        answer = str(index_from_end)
        return task_def, answer
    
    # If substring didn't occur N times, return -1
    task_def = f"IC{substring}{occurrence}"
    answer = "-1"
    return task_def, answer

# Task 5 (T): Token value at index N before the end token
def token_at_index_task(pcfg_string: str, index: int = 40) -> Tuple[str, str]:
    """Get the token value at index N before the end token."""
    if index <= len(pcfg_string):
        # Index from the end
        token = pcfg_string[-index]
        task_def = f"TNULL{index}"
        answer = token
        return task_def, answer
    
    # If index is out of bounds, return empty
    task_def = f"TNULL{index}"
    answer = ""
    return task_def, answer


# Build dataset

def _tokenize_with_numbers(text: str) -> List[str]:
    """Tokenize text so contiguous digits form a single token.
    
    Examples:
      'Ca40' -> ['C', 'a', '40']
      'ICaa6' -> ['I', 'C', 'a', 'a', '6']
      '-1' -> ['-1']
    """
    tokens = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isdigit():
            j = i + 1
            while j < len(text) and text[j].isdigit():
                j += 1
            tokens.append(text[i:j])
            i = j
        elif ch == '-' and i + 1 < len(text) and text[i + 1].isdigit():
            j = i + 2
            while j < len(text) and text[j].isdigit():
                j += 1
            tokens.append(text[i:j])
            i = j
        else:
            tokens.append(ch)
            i += 1
    return tokens

def format_example(pcfg_string: str, task_def: str, answer: str) -> List[str]:
    """Format a training example as a list of string tokens.
    
    Format: [SOS] Task_Def... [SOT] PCFG_chars... [EOT] [ART] Answer_tokens... [EOS]
    """
    tokens = ['[SOS]']
    tokens.extend(_tokenize_with_numbers(task_def))
    tokens.append('[SOT]')
    tokens.extend(list(pcfg_string))
    tokens.append('[EOT]')
    tokens.append('[ART]')
    tokens.extend(_tokenize_with_numbers(answer))
    tokens.append('[EOS]')
    return tokens

class CharTokenizer:
    """Token-level tokenizer: maps string tokens to integer IDs."""
    
    def __init__(self):
        NUMERIC_TOKENS = [str(i) for i in range(0, 251)] + ['-1']
        vocab_tokens = [
                        # task family tokens
                        'C', 'I', 'T', 'N', 'U', 'L',
                        # operands and PCFG terminals
                        'a', 'b', 'c',
                        # numeric tokens used in task defs/answers
                        *NUMERIC_TOKENS,
                                        ]
        special_tokens = {
                    'SOS': '[SOS]',    # Start of sequence
                    'SOT': '[SOT]',    # Start of text (PCFG string)
                    'EOT': '[EOT]',    # End of text
                    'ART': '[ART]',    # Answer region token
                    'EOS': '[EOS]',    # End of sequence
                    'PAD': '[PAD]',    # Padding token
                }
        
        self.special_tokens = special_tokens
        
        # Build vocab: special tokens first, then observed tokens
        vocab_items = list(special_tokens.values()) + vocab_tokens
        
        # Remove duplicates while preserving order
        seen = set()
        vocab = []
        for item in vocab_items:
            if item not in seen:
                seen.add(item)
                vocab.append(item)
        
        self.vocab = vocab
        self.tok2idx = {tok: idx for idx, tok in enumerate(vocab)}
        self.idx2tok = {idx: tok for idx, tok in enumerate(vocab)}
        self.vocab_size = len(vocab)
        
        # Store special token IDs
        self.pad_id = self.tok2idx[special_tokens['PAD']]
        self.eos_id = self.tok2idx[special_tokens['EOS']]
        self.art_id = self.tok2idx[special_tokens['ART']]
        
        print(f"Tokenizer vocabulary size: {self.vocab_size}")
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert a list of string tokens to a list of integer IDs."""
        ids = []
        for tok in tokens:
            if tok in self.tok2idx:
                ids.append(self.tok2idx[tok])
            else:
                print(f"Warning: Unknown token '{tok}'")
        return ids
    
class PCFGDataset(Dataset):
    """Dataset for PCFG examples (token-list format)."""
    
    def __init__(self, examples: List[List[str]], tokenizer: CharTokenizer, max_length: int = 512, mask_answer_only: bool = True):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_answer_only = mask_answer_only
        
        # Pre-encode all examples
        self.encoded_examples = []
        self.answer_positions = []  # Track where the answer tokens are
        
        art_tok = '[ART]'
        eos_tok = '[EOS]'
        
        for token_list in examples:
            ids = tokenizer.encode(token_list)
            
            # Find answer positions directly from the token list
            try:
                art_idx = token_list.index(art_tok)
                eos_idx = token_list.index(eos_tok)
                # Answer tokens are between ART and EOS (exclusive of both)
                answer_pos = list(range(art_idx + 1, eos_idx))
            except ValueError:
                answer_pos = []
            
            # Truncate if necessary
            if len(ids) > max_length:
                ids = ids[:max_length]
                answer_pos = [p for p in answer_pos if p < max_length]
            
            self.encoded_examples.append(ids)
            self.answer_positions.append(answer_pos)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.encoded_examples[idx]
        answer_pos = self.answer_positions[idx]
        
        # Create input (all tokens except last) and target (all tokens except first)
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        
        # Adjust answer positions for the shift
        answer_pos_shifted = [p - 1 for p in answer_pos if p > 0 and p - 1 < len(target_ids)]
        
        if self.mask_answer_only:
            # Create loss mask: -100 for non-answer positions
            loss_mask = torch.full((len(target_ids),), -100, dtype=torch.long)
            for pos in answer_pos_shifted:
                if 0 <= pos < len(target_ids):
                    loss_mask[pos] = target_ids[pos]
            target_tensor = loss_mask
        else:
            # Full next-token prediction targets
            target_tensor = torch.tensor(target_ids, dtype=torch.long)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': target_tensor,
            'answer_positions': answer_pos_shifted,
        }
