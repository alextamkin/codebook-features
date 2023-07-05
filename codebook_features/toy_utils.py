"""Util functions for the toy model."""
import itertools

import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from tqdm import tqdm


def get_codebook_info_from_hook_key(hook_key):
    """Get the layer, component and head from the hook key."""
    parts = hook_key.split(".")
    layer = int(parts[1])
    attn_or_mlp = parts[2]
    if attn_or_mlp == "attn":
        head = int(parts[5])
    else:
        head = None
    return layer, attn_or_mlp, head


def find_code_changes(cache1, cache2):
    """Find the code changes between two caches."""
    for k in cache1.keys():
        if "codebook" in k:
            c1 = cache1[k][0, -1]
            c2 = cache2[k][0, -1]
            if not torch.all(c1 == c2):
                print(get_codebook_info_from_hook_key(k), c1.tolist(), c2.tolist())


def start_state_activations(
    state, cb_model, automata, cb_at="attn", layer=0, ccb_num=3
):
    """Get all the states on which the code(s) at the given component and state are activated."""
    cb_model.reset_hook_kwargs()
    base_input = cb_model.to_tokens(automata.traj_to_str([state]), prepend_bos=True)
    base_input = base_input.to("cuda")
    base_logits, base_cache = cb_model.run_with_cache(base_input)
    if cb_at == "attn":
        cache_str = (
            f"blocks.{layer}.attn.codebook_layer.codebook.{ccb_num}.hook_codebook_ids"
        )
        indices = base_cache[cache_str][0, -1].tolist()
    else:
        cache_str = f"blocks.{layer}.mlp.codebook_layer.hook_codebook_ids"
        indices = base_cache[cache_str][0, -1].tolist()
    print("code", indices)
    states_activated_on = []

    for start_state in range(automata.N):
        mod_input = cb_model.to_tokens(
            automata.traj_to_str([start_state]), prepend_bos=True
        ).to("cuda")
        mod_logits, mod_cache = cb_model.run_with_cache(mod_input)
        mod_indices = mod_cache[cache_str][0, -1].tolist()
        if indices == mod_indices:
            states_activated_on.append(start_state)
    return states_activated_on


def valid_input(input: str, automata):
    """Check if the input is a valid string that can be generated by the automata."""
    assert len(input) <= automata.digits + 1
    if len(input) < automata.digits + 1:
        return True
    state = automata.seq_to_traj(input)[0][0]
    ns_start_token = input[-1]
    possible_next_states = automata.nbrs(state)
    possible_start_tokens = [automata.token_repr(s)[0] for s in possible_next_states]
    return ns_start_token in possible_start_tokens


def partition_input_on_codebook(
    cb_model,
    automata,
    cb_at,
    layer,
    ccb_num,
    input_len=2,
):
    """Partition the input pattern based on the codes at the specified component."""
    if cb_at == "attn":
        cache_str = (
            f"blocks.{layer}.attn.codebook_layer.codebook.{ccb_num}.hook_codebook_ids"
        )
    else:
        cache_str = f"blocks.{layer}.mlp.codebook_layer.hook_codebook_ids"

    partition = {}
    chars = [str(c) for c in range(automata.representation_base)]
    input_range = itertools.product(chars, repeat=input_len)
    for inp_tuple in tqdm(input_range):
        inp = "".join(inp_tuple)
        if not valid_input(inp, automata):
            continue
        mod_input = cb_model.to_tokens(inp, prepend_bos=True).to("cuda")
        mod_logits, mod_cache = cb_model.run_with_cache(mod_input)
        mod_indices = mod_cache[cache_str][0, -1].tolist()
        for mod_index in mod_indices:
            if mod_index not in partition:
                partition[mod_index] = []
            partition[mod_index].append(inp)
    return partition


def get_next_state_probs(state, model, automata, fwd_hooks=None, prepend_bos=True):
    """Get the top next state probabilities given by the model."""
    if isinstance(state, int):
        state_str = automata.traj_to_str([state])
        state = model.to_tokens(state_str, prepend_bos=prepend_bos).to("cuda")
    elif not isinstance(state, torch.Tensor):
        raise ValueError("state must be an int or a tensor of state inputs")

    if fwd_hooks is not None:

        def model_run(x):
            return model.run_with_hooks(x, fwd_hooks=fwd_hooks)

    else:

        def model_run(x):
            return model(x)

    next_state_probs = torch.zeros((state.shape[0], automata.N)).to("cuda")
    base = automata.representation_base
    for next_token in range(base):
        next_token_input = F.pad(state, (0, 1), value=next_token).to("cuda")
        next_token_logits = model_run(next_token_input)
        if isinstance(next_token_logits, dict):
            next_token_logits = next_token_logits["logits"]
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        next_state_prob = (
            next_token_probs[:, -2, next_token].unsqueeze(-1)
            * next_token_probs[:, -1, :base]
        )
        next_state_probs[
            :, next_token * base : (next_token + 1) * base
        ] = next_state_prob

    # filter next_state_probs to only include the top `edges`
    next_state_probs, next_state_preds = torch.topk(
        next_state_probs, automata.edges, dim=-1, sorted=True
    )
    return next_state_preds, next_state_probs


def correct_next_state_probs(state, next_state_probs, automata, print_info=""):
    """Get the accuracy of the next state probabilities."""
    if isinstance(next_state_probs, tuple):
        next_states = next_state_probs[0]
    elif isinstance(next_state_probs, torch.Tensor):
        next_states = next_state_probs
    else:
        raise ValueError("next_state_probs must be a tensor or tuple")

    if isinstance(state, list):
        state = [int(s) for s in state]
    elif isinstance(state, int):
        state = [state] * next_states.shape[0]
    elif not isinstance(state, list):
        raise ValueError("state must be an int or a list of ints or a tensor.")

    next_states_pred = torch.zeros((next_states.shape[0], automata.N), dtype=bool).to(
        "cuda"
    )
    next_states_pred.scatter_(1, next_states, True)

    actual_next_states = automata.transition_matrix[state, :] > 0
    common = actual_next_states * next_states_pred.cpu().numpy()
    accuracy = common.sum(axis=-1) / automata.edges
    if "i" in print_info:
        incorrect_transitions = next_states_pred - common
        for i, s in enumerate(state):
            print(
                f"incorrect transitions: {s} ->  {incorrect_transitions[i].nonzero().tolist()}"
            )
    if "c" in print_info:
        print(f"Correct transitions: {state} ->  {common}")
    return accuracy


def first_transition_accuracy(model, automata, fwd_hooks=None, prepend_bos=True):
    """Get the average accuracy of the first transition."""
    avg_acc = 0
    for state in tqdm(range(automata.N)):
        nsp = get_next_state_probs(state, model, automata, fwd_hooks, prepend_bos)[0]
        acc = correct_next_state_probs(state, nsp, automata)
        avg_acc += acc
    avg_acc /= automata.N
    return avg_acc


def plot_js_div(
    code_groups_for_all_comps,
    layer,
    cb_at,
    ccb_num,
    js_divs_state_pairs,
    show_plot=False,
    image_name_prefix=None,
):
    """Plot the histogram of JSD between all pairs of states and given code groups."""
    group_js_divs = {}
    code_groups = code_groups_for_all_comps[(layer, cb_at, ccb_num)]
    for code, grouped_states in code_groups.items():
        if len(grouped_states) < 2:
            continue
        group_js_divs[code] = []
        for ia, sa in enumerate(grouped_states):
            for sb in grouped_states[ia + 1 :]:
                group_js_divs[code].append(js_divs_state_pairs[(sa, sb)])

    avg_group_js_divs = {
        code: sum(grp_js_divs) / len(grp_js_divs)
        for code, grp_js_divs in group_js_divs.items()
    }

    js_values = list(js_divs_state_pairs.values())
    input_len = len(list(js_divs_state_pairs.keys())[0][0])
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=js_values,
            nbinsx=100,
            histnorm="probability",
            name=f"JS Divergence for {'bigram' if input_len == 2 else 'trigram'} inputs",
        )
    )
    avg_group_js_divs_values = list(avg_group_js_divs.values())
    fig.add_trace(
        go.Histogram(
            x=avg_group_js_divs_values,
            nbinsx=100,
            histnorm="probability",
            name="Code Group JS Divergence",
        )
    )
    avg_js_div = sum(js_values) / len(js_values)
    comp_title = f"Layer {layer} {'Attn Head ' if cb_at == 'attn' else 'MLP'}{ccb_num if ccb_num is not None else ''}"
    fig.update_layout(
        barmode="group", title=f"JS Divergence for {comp_title} (avg={avg_js_div:.3f})"
    )
    if show_plot:
        fig.show()
    if image_name_prefix is not None:
        fig.write_image(
            f"{image_name_prefix}_js_div_{layer}_{cb_at}{ccb_num if ccb_num is not None else ''}.png"
        )


def get_layers_from_patching_str(patching):
    """Get the layers from the patching string."""
    layers = patching.split("_")[0].split(",")
    layers = [int(layer[1:]) for layer in layers]
    return layers


def clean_patching_name(patching):
    """Clean the patching name."""
    if patching == "none":
        return "None"
    layers = patching.split("_")[0].split(",")
    cb_at = patching.split("_")[1:]
    cb_at_map = {"attn": "Attn", "mlp": "MLP"}
    layers_map = {"l": "L", "aLL": "All"}
    clean_layers = []
    for layer in layers:
        clean_layers.append(layer)
        for k, v in layers_map.items():
            clean_layers[-1] = clean_layers[-1].replace(k, v)
    clean_patching = ", ".join(clean_layers)
    clean_cb_at = []
    for i_cb_at in cb_at:
        clean_cb_at.append(i_cb_at)
        for k, v in cb_at_map.items():
            clean_cb_at[-1] = clean_cb_at[-1].replace(k, v)
    clean_patching += " " + ", ".join(clean_cb_at)
    return clean_patching
