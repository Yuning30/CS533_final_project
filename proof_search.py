import tqdm
import torch
import json
import random
import math
import sys
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# print("recursion limit", sys.getrecursionlimit())
# sys.setrecursionlimit(5000)
# print("recursion limit", sys.getrecursionlimit())

random.seed(1)
path = "entailment_bank/data/public_dataset/entailment_trees_emnlp2021_data_v2/dataset/task_1/"

t5_tokenizer = T5Tokenizer.from_pretrained("t5-large")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

device = 'cuda:1'

class Node:
    def __init__(self, type_in=None, score_in=None, sent_in=None):
        self.in_bound_edges = []
        self.out_bound_edges = []
        self.type = type_in
        self.score = score_in
        self.sent = sent_in 


def proof_search(one_task, prover, verifier):
    graph_nodes, sent_to_node_mapping = generate_greedy_and_initialize_graph(one_task, prover, verifier)
    if graph_nodes is None:
        return ""
    explored = set()

    while True:
        # import pdb
        # pdb.set_trace()
        sampled_partial_proof_str, sent_to_symbol_mapping = sample_new(one_task, graph_nodes, sent_to_node_mapping, explored)
        if sampled_partial_proof_str is None:
            # no new proof found in a reasonable time
            print("EARLY TERMINATION")
            break
        explored.add(sampled_partial_proof_str)
        print("sampled partial proof", sampled_partial_proof_str)

        proof_step_list, p_scrs = prover_generate(one_task, prover, sampled_partial_proof_str, sent_to_symbol_mapping)
         
        v_scrs = verifier_verify(one_task, verifier, proof_step_list)

        p_scrs = np.array(p_scrs)
        v_scrs = np.array(v_scrs)
        scrs = (p_scrs + v_scrs) / 2

        print(proof_step_list)
        print(scrs)

        # import pdb
        # pdb.set_trace()
        graph_nodes, updated = update_graph(one_task, graph_nodes, proof_step_list, scrs)
        if not updated:
            break
    
    return extract_proof(one_task, graph_nodes)

def extract_proof(one_task, graph_nodes):
    # get hypothesis node and all of its predecessors
    print([node.type for node in graph_nodes])
    for node in graph_nodes:
        if node.type == "h":
            hypothesis_node = node

    proof_nodes = []
    nodes_to_be_added = [hypothesis_node]

    while len(nodes_to_be_added) > 0:
        node = nodes_to_be_added[0]
        proof_nodes.append(node)
        nodes_to_be_added.remove(node)

        premise_nodes = node.in_bound_edges[0].in_bound_edges
        for p_node in premise_nodes:
            if p_node.type == "C":
                continue
            assert (p_node.type == "I")
            if p_node not in nodes_to_be_added:
                nodes_to_be_added.append(p_node)

    # get the final proof string
    sent_to_symbol = {}
    for symbol, sent in one_task["meta"]["triples"].items():
        sent_to_symbol[sent] = symbol

    proof_str = ""
    int_idx = 1
    #import pdb
    #pdb.set_trace()
    while len(proof_nodes) > 0:
        for idx, node in enumerate(proof_nodes):
            premise_sents = [premise_node.sent for premise_node in node.in_bound_edges[0].in_bound_edges]
            all_exist = all([sent in sent_to_symbol for sent in premise_sents])
            if all_exist:
                # convert this step to string
                premise_symbol = [sent_to_symbol[sent] for sent in premise_sents]
                if node.type == "h":
                    step_str = f" {' & '.join(premise_symbol)} -> hypothesis;"
                elif node.type == "I":
                    int_x = f"int{int_idx}"
                    step_str = f" {' & '.join(premise_symbol)} -> {int_x}: {node.sent};"
                    sent_to_symbol[node.sent] = int_x
                    int_idx += 1
                else:
                    assert False

                proof_str += step_str
                proof_nodes.remove(node)
                break

    
    print("final proof str:", proof_str)
    return proof_str



def update_graph(one_task, graph_nodes, proof_step_list, scrs):
    sent_to_node_mapping = {}
    for node in graph_nodes:
        if node.type in ["h", "I", "C"]:
            sent_to_node_mapping[node.sent] = node

    updated = False
    for idx, (premise_sents_list, conclusion_sent, final_step) in enumerate(proof_step_list):
        conclusion_sent = one_task['hypothesis'] if final_step else conclusion_sent
        if conclusion_sent in sent_to_node_mapping:
            temp_S_node = Node(type_in="S", score_in=scrs[idx])
            temp_I_node = Node(type_in="h" if final_step else "I", score_in=min(scrs[idx], min([sent_to_node_mapping[x].score for x in premise_sents_list])), sent_in=conclusion_sent)

            if temp_I_node.score > sent_to_node_mapping[conclusion_sent].score:
                print("update for better score for sentence:", conclusion_sent)
                # find a better proof for current sent, update the graph
                old_I_node = sent_to_node_mapping[conclusion_sent]
                old_S_node = old_I_node.in_bound_edges[0]

                for premise_node in old_S_node.in_bound_edges:
                    premise_node.out_bound_edges.remove(old_S_node)


                temp_S_node.out_bound_edges = [temp_I_node]
                temp_I_node.in_bound_edges = [temp_S_node]

                temp_S_node.in_bound_edges = [sent_to_node_mapping[x] for x in premise_sents_list]
                for node in temp_S_node.in_bound_edges:
                    node.out_bound_edges.append(temp_S_node)

                # replace old_I_node with temp_I_node
                temp_I_node.out_bound_edges = old_I_node.out_bound_edges

                graph_nodes.remove(old_S_node)
                graph_nodes.remove(old_I_node)
                graph_nodes.append(temp_S_node)
                graph_nodes.append(temp_I_node)
                sent_to_node_mapping[temp_I_node.sent] = temp_I_node

                score_to_change_list = []
                for successor in temp_I_node.out_bound_edges:
                    successor.in_bound_edges.remove(old_I_node)
                    successor.in_bound_edges.append(temp_I_node)
                    score_to_change_list.append(successor)

                # propogate score update to all possible successors
                while len(score_to_change_list) > 0:
                    successor = score_to_change_list[0]
                    score_to_change_list.pop(0)
                    if successor.type == "S":
                        pass
                    elif successor.type == "I" or successor.type == "h":
                        successor.score = min(successor.in_bound_edges[0].score, min([premise.score for premise in successor.in_bound_edges[0].in_bound_edges]))

                    # add successor's successor
                    for node in successor.out_bound_edges:
                        score_to_change_list.append(node)

                updated = True
            else:
                print("no update for sentence:", conclusion_sent)
        else:
            # create a new node for this step
            print("add node for sentence", conclusion_sent)
            assert (not final_step) 
            S_node = Node(type_in="S", score_in=scrs[idx])
            I_node = Node(type_in="I", score_in=min(scrs[idx], min([sent_to_node_mapping[x].score for x in premise_sents_list])), sent_in=conclusion_sent)

            S_node.out_bound_edges = [I_node]
            I_node.in_bound_edges = [S_node]

            S_node.in_bound_edges = [sent_to_node_mapping[x] for x in premise_sents_list]
            for node in S_node.in_bound_edges:
                node.out_bound_edges.append(S_node)

            graph_nodes.append(S_node)
            graph_nodes.append(I_node)

            updated = True


    return graph_nodes, updated



def verifier_verify(one_task, verifier, proof_step_list):
    score_list = []
    for premise_sents_list, conclusion_sent, final_step in proof_step_list:
        premise_sents = ". ".join(premise_sents_list)
        conclusion_sent = one_task["hypothesis"] if final_step else conclusion_sent

        step_encoding = roberta_tokenizer(text=premise_sents, text_pair=conclusion_sent, padding="longest", max_length=512, truncation=True, return_tensors="pt")
        input_ids, attention_mask = step_encoding.input_ids.to(device), step_encoding.attention_mask.to(device)
        with torch.no_grad():
            verifier_logits = verifier(input_ids=input_ids, attention_mask=attention_mask).logits
            verifier_score = torch.sigmoid(verifier_logits).squeeze().item()

        score_list.append(verifier_score)

    return score_list



def prover_generate(one_task, prover, partial_proof_str, sent_to_symbol_mapping):
    symbol_to_sent_mapping = {}
    for sent, symbol in sent_to_symbol_mapping.items():
        symbol_to_sent_mapping[symbol] = sent

    hypothesis_str = f"$hypothesis$ = {one_task['hypothesis']};"
    fact_str = f"$fact$ = {one_task['context']};"

    input_str = f"{hypothesis_str} {fact_str} {partial_proof_str}"

    input_ids = t5_tokenizer(input_str, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = prover.generate(input_ids, output_scores=True, max_length=128, return_dict_in_generate=True, num_beams=4, num_return_sequences=2, length_penalty=0.0)
        # import pdb
        # pdb.set_trace()

    proof_step_list = []
    score_list = []
    for i in range(len(outputs.sequences)):
        proof_step = t5_tokenizer.decode(outputs.sequences[i], skip_special_tokens=True)
        print("generated proof step:", proof_step)
        if "->" not in proof_step:
            print("invalid proof step: without ->")
            continue
        premises, conclusion, final_step = parse_proof_step(proof_step)

        # # prover_score = 1
        # for logits in outputs.scores[i]:
            # logits = logits[0]
            # current_score = logits.softmax(dim=0)[logits.argmax()].item()
            # prover_score *= current_score
        prover_score = math.exp(outputs.sequences_scores[i])

        try:
            premise_sents = [symbol_to_sent_mapping[symbol] for symbol in premises]

            if len(premise_sents) > len(set(premise_sents)):
                print("invalid proof step: sent/int used more than 1 time")
                continue

            if not final_step:
                int_premises = list(filter(lambda x: x.startswith("int"), premises))
                if conclusion[1] in [symbol_to_sent_mapping[x] for x in int_premises]:
                    print("invalid proof step: copy int to int")
                    continue

                if conclusion[1] == one_task['hypothesis']:
                    print("invalid proof step: output hypothesis at not final step")
                    continue
        except:
            # invalid proof step using some unknown symnbols
            print("invalid proof step: use unknown symbols")
            continue

        

        conclusion_sent = "hypothesis" if final_step else conclusion[1]
        proof_step_list.append((premise_sents, conclusion_sent, final_step))
        score_list.append(prover_score)

    return proof_step_list, score_list

def sample_new(one_task, graph_nodes, sent_to_node_mapping, explored):
    iters = 1000
    for _ in range(iters):
        partial_str, sent_to_symbol_mapping = sample_new_one(one_task, graph_nodes, sent_to_node_mapping)
        if partial_str not in explored:
            # find a new partial proof
            return partial_str, sent_to_symbol_mapping

    return None, None


def sample_new_one(one_task, graph_nodes, sent_to_node_mapping):
    # filter out the I nodes
    I_nodes = []
    for node in graph_nodes:
        if node.type == "I":
            I_nodes.append(node)
    
    # do a topological sort for I_nodes
    sorted_nodes = []
    nodes_to_be_added = I_nodes

    while len(nodes_to_be_added) > 0:
        for idx, node in enumerate(nodes_to_be_added):
            all_added = True
            for S_node in node.out_bound_edges:
                assert (S_node.type == "S")
                for next_node in S_node.out_bound_edges:
                    assert (next_node.type in ["I", "h"])
                    if next_node.type == "I":
                        # check if this node is already added in sorted list
                        if next_node not in sorted_nodes:
                            all_added = False

            if all_added:
                sorted_nodes.append(node)
                nodes_to_be_added.remove(node)
                break

    # here we have a topological sort for the graph
    partial_proof_nodes = []
    required_to_add_nodes = []
    for idx, node in enumerate(sorted_nodes):
        if node in required_to_add_nodes:
            # this node is required to be added as other node's predecessor
            partial_proof_nodes.append(node)
            required_to_add_nodes.remove(node)

            # add all predecessors to required_to_add_nodes
            assert (len(node.in_bound_edges) == 1)
            for pred in node.in_bound_edges[0].in_bound_edges:
                assert (pred.type in ["I", "C"])
                if pred.type == "I" and pred not in required_to_add_nodes:
                    required_to_add_nodes.append(pred)
        else:
            choice = random.choice([0, 1])
            if choice == 1:
                partial_proof_nodes.append(node)
                # add all predecessors to required_to_add_nodes
                assert (len(node.in_bound_edges) == 1)
                for pred in node.in_bound_edges[0].in_bound_edges:
                    assert (pred.type in ["I", "C"])
                    if pred.type == "I" and pred not in required_to_add_nodes:
                        required_to_add_nodes.append(pred)
    
    # construct partial proof string
    sent_to_symbol_mapping = {}
    for key, value in one_task["meta"]["triples"].items():
        sent_to_symbol_mapping[value] = key

    partial_proof_nodes = list(reversed(partial_proof_nodes))
    partial_str = "$partial_proof$ ="
    for idx, node in enumerate(partial_proof_nodes):
        assert (len(node.in_bound_edges) == 1)
        premises = node.in_bound_edges[0].in_bound_edges
        premise_symbols = []
        for premise_node in premises:
            if premise_node.type in ["I", "C"]:
                premise_symbols.append(sent_to_symbol_mapping[premise_node.sent])
            else:
                assert False

        symbol = f"int{idx + 1}"
        step_str = f" {' & '.join(premise_symbols)} -> {symbol}: {node.sent};"
        partial_str += step_str

        sent_to_symbol_mapping[node.sent] = symbol

    return partial_str, sent_to_symbol_mapping

def generate_greedy_and_initialize_graph(one_task, prover, verifier):
    """
        return a proof for the task generated by the prover using a greedy way
    """
    # import pdb
    # pdb.set_trace()
    greedy_proof = []
    graph_nodes = []

    symbol_to_sent_mapping = {}
    sent_to_node_mapping = {}

    
    symbol_to_sent_mapping["hypothesis"] = one_task["hypothesis"]
    sent_to_node_mapping[one_task["hypothesis"]] = Node(type_in="h", score_in=None, sent_in=one_task["hypothesis"])
    graph_nodes.append(sent_to_node_mapping[one_task["hypothesis"]])
    
    for key, value in one_task["meta"]["triples"].items():
        symbol_to_sent_mapping[key] = value
        sent_to_node_mapping[value] = Node(type_in="C", score_in=1.0, sent_in=value)
        graph_nodes.append(sent_to_node_mapping[value])

    hypothesis_str = f"$hypothesis$ = {one_task['hypothesis']};"
    fact_str = f"$fact$ = {one_task['context']};"

    # initial partial proof is empty
    partial_proofs_str = "$partial_proof$ ="

    # generate one proof greadily
    input_str = f"{hypothesis_str} {fact_str} {partial_proofs_str}"

    while True:
        input_ids = t5_tokenizer(input_str, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        with torch.no_grad():
            outputs = prover.generate(input_ids, output_scores=True, max_length=128, return_dict_in_generate=True)
        proof_step = t5_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        # parse the output
        valid = True
        if "->" not in proof_step:
            valid = False
        else:
            premises, conclusion, final_step = parse_proof_step(proof_step)

            if not all([x in symbol_to_sent_mapping for x in premises]):
                valid = False
                    
            if not final_step:
                # invalid case: int_x & ... -> int_y (directly copy a int_x to be the result but give a different number)
                int_premises = list(filter(lambda x: x.startswith("int"), premises))
                if conclusion[1] in [symbol_to_sent_mapping[x] for x in int_premises]:
                    valid = False

                # invalid case: output hypothesis as one int result
                if conclusion[1] == one_task['hypothesis']:
                    valid = False
        
        if valid:
            greedy_proof.append((premises, conclusion, final_step))
            if not final_step:
                symbol_to_sent_mapping[conclusion[0]] = conclusion[1]

            # calculate prover scores
            prover_score = 1
            for logits in outputs.scores:
                logits = logits[0]
                current_score = logits.softmax(dim=0)[logits.argmax()].item()
                prover_score *= current_score
        else:
            print("invalid step during greedy search")
            # import pdb
            # pdb.set_trace()
            outputs = prover.generate(input_ids, output_scores=True, max_length=128, return_dict_in_generate=True, num_beams=6, num_return_sequences=6, length_penalty=0.0)
            for i in range(len(outputs.sequences)):
                proof_step = t5_tokenizer.decode(outputs.sequences[i], skip_special_tokens=True)
                print("generated proof step:", proof_step)
                if "->" not in proof_step:
                    print("invalid proof step: not -> symbol")
                    continue
                premises, conclusion, final_step = parse_proof_step(proof_step)

                if not all([x in symbol_to_sent_mapping for x in premises]):
                    # uses unknown sents or ints
                    continue

                if not final_step:
                    if conclusion[1] not in [symbol_to_sent_mapping[x] for x in premises] and conclusion[1] != one_task['hypothesis']:
                        valid = True
                        prover_score = math.exp(outputs.sequences_scores[i])
                        greedy_proof.append((premises, conclusion, final_step))
                        symbol_to_sent_mapping[conclusion[0]] = conclusion[1]
                        break
                else:
                    valid = True
                    prover_score = math.exp(outputs.sequences_scores[i])
                    greedy_proof.append((premises, conclusion, final_step))
                    assert (conclusion[0] == 'hypothesis')
                    assert (symbol_to_sent_mapping['hypothesis'] == one_task['hypothesis'])
                    break
            if valid == False:
                print("invalid again")
                return None, None

        # calculate verifier scores, steps needs to be valid at this point
        premise_sents = ". ".join([symbol_to_sent_mapping[x] for x in premises])
        if final_step:
            conclusion_sent = symbol_to_sent_mapping["hypothesis"]
        else:
            conclusion_sent = conclusion[1]

        step_encoding = roberta_tokenizer(text=premise_sents, text_pair=conclusion_sent, padding="longest", max_length=512, truncation=True, return_tensors="pt") 
        input_ids, attention_mask = step_encoding.input_ids.to(device), step_encoding.attention_mask.to(device)
        with torch.no_grad():
            verifier_logits = verifier(input_ids=input_ids, attention_mask=attention_mask).logits
            verifier_score = torch.sigmoid(verifier_logits).squeeze()

        # update the graph
        S_node = Node(type_in="S", score_in=(prover_score + verifier_score) / 2)
        S_node.in_bound_edges = [sent_to_node_mapping[symbol_to_sent_mapping[x]] for x in premises]
        for premise_node in S_node.in_bound_edges:
            premise_node.out_bound_edges.append(S_node)
        graph_nodes.append(S_node)

        if final_step:
            # update scores for hypothesis
            sent_to_node_mapping[one_task["hypothesis"]].in_bound_edges = [S_node]
            sent_to_node_mapping[one_task["hypothesis"]].score = min(S_node.score, min([sent_to_node_mapping[symbol_to_sent_mapping[x]].score for x in premises]))

            S_node.out_bound_edges = [sent_to_node_mapping[one_task["hypothesis"]]]
            break
        else:
            I_node = Node(type_in="I", score_in=min(S_node.score, min([sent_to_node_mapping[symbol_to_sent_mapping[x]].score for x in premises])), sent_in=conclusion[1])
            I_node.in_bound_edges = [S_node]

            S_node.out_bound_edges = [I_node]

            sent_to_node_mapping[conclusion[1]] = I_node
            graph_nodes.append(I_node)

        # update the input string
        input_str = f"{input_str} {proof_step}"

    # return the greedy proof and its scores
    return graph_nodes, sent_to_node_mapping

# graph is just represented by a collection of nodes

def parse_proof_step(proof_step):
    # proof step should be a sentence from the prover
    print(proof_step)
    lhs, rhs = proof_step.split("->")
    

    premises = lhs.split("&")
    for i in range(0, len(premises)):
        premises[i] = premises[i].strip()

    final_step = False
    conclusion = []
    if ":" in rhs:
        # rhs should be an int sentence
        rhs = rhs.split(":")
        conclusion.append(rhs[0].strip())
        conclusion.append(rhs[1].strip()[0:-1])
    else:
        # rhs should be hypothesis
        conclusion.append(rhs.strip()[0:-1])
        assert(conclusion[0] == "hypothesis")
        final_step = True

    return premises, conclusion, final_step


def initialize_graph(proof, scores):
    graphs = []





def eval_task(file_path, prover, verifier):
    # import pdb
    # pdb.set_trace()
    with open(file_path, "r") as f:
        with open("result.txt", "w") as w:
            lines = f.readlines()
            for idx, line in tqdm.tqdm(enumerate(lines)):
                # if idx == 0:
                    # import pdb
                    # pdb.set_trace()
                one_task = json.loads(line)
                print(f"--------- test {idx} -----------\n")
                proof_str = proof_search(one_task, prover, verifier)
                print("")
                # w.write(f"round truth proof\t\t: {one_task['proof']}\n")
                # w.write(f"our proof\t\t: {proof_str}\n")
                w.write(f"$proof$ ={proof_str}\n")
            

if __name__ == "__main__":
    prover = T5ForConditionalGeneration.from_pretrained("./saved_model/T5_large_epoch_2/").to(device)
    verifier = RobertaForSequenceClassification.from_pretrained("./saved_model/roberta_large_epoch_3/").to(device)
    
    prover.eval()
    verifier.eval()

    eval_task(path + "test.jsonl", prover, verifier)
