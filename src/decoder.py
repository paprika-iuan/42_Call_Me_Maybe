import numpy as np

RNG = np.random.default_rng()


def discard_bad_delimiters(logits: np.ndarray,
                           bad_delim: list[list[int]]) -> None:
    if bad_delim:
        first_tokens = [d[0] for d in bad_delim if d]
        logits[first_tokens] = float("-inf")


def apply_temperature(tempVar: float, logits: np.ndarray) -> None:
    if tempVar > 0:
        logits /= tempVar


def apply_topk(k: int, logits: np.ndarray) -> None:
    n_logits = len(logits)
    if 0 < k < n_logits:
        indexes_to_cut = n_logits - k
        partitioned = np.argpartition(logits, indexes_to_cut)
        discarted = partitioned[:indexes_to_cut]

        logits[discarted] = float('-inf')


def apply_topp(p: float, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_indexes = np.where(logits > float('-inf'))[0]
    valid_logits = logits[valid_indexes]

    sorted_local_indexes = np.argsort(valid_logits)[::-1]
    sorted_logits = valid_logits[sorted_local_indexes]
    sorted_global_indexes = valid_indexes[sorted_local_indexes]

    max_logit = np.max(sorted_logits)
    probs = np.exp(sorted_logits - max_logit)
    probs /= np.sum(probs)

    if p < 1.0:
        cumsum = np.cumsum(probs)
        indexes_to_remove = cumsum > p
        indexes_to_remove[1:] = indexes_to_remove[:-1]
        indexes_to_remove[0] = False

        valid_probs = probs[~indexes_to_remove]
        valid_global_indexes = sorted_global_indexes[~indexes_to_remove]
        valid_probs /= np.sum(valid_probs)
    else:
        valid_probs = probs
        valid_global_indexes = sorted_global_indexes

    return valid_probs, valid_global_indexes


def apply_multinominal(probs: np.ndarray, global_indexes: np.ndarray) -> int:
    cumsum = np.cumsum(probs)
    closer_index = int(np.searchsorted(cumsum, RNG.random()))
    max_index = len(global_indexes) - 1
    closer_index = min(closer_index, max_index)
    return int(global_indexes[closer_index])


def constrained_decoding(logits: np.ndarray,
                         tmp: float,
                         topk: int,
                         topp: float,
                         bad_delim: list[list[int]]) -> int:
    if tmp <= 0:
        discard_bad_delimiters(logits, bad_delim)
        return int(np.argmax(logits))
    discard_bad_delimiters(logits, bad_delim)
    apply_temperature(tmp, logits)
    apply_topk(topk, logits)
    valid_probs, valid_global_indexes = apply_topp(topp, logits)
    return apply_multinominal(valid_probs, valid_global_indexes)
