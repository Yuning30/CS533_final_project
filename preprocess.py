import os
import json
import random
import torch
import rank_bm25
import numpy as np

import entailment_bank.utils.proof_utils as proof_utils
path = "entailment_bank/data/public_dataset/entailment_trees_emnlp2021_data_v2/dataset/task_1/"

with open(path + "train.jsonl", "r") as f:
    lines = f.readlines()
    one_task = json.loads(lines[0])

    # for line in lines:
    #     one_task = json.loads(line)
    #     break

for key, value in one_task.items():
    print(key)
    print(value)
    print('---------')

proof_utils.parse_entailment_step_proof(one_task["proof"], one_task, print_flag=True)

def preprocess(file_path):
    """
        read the data file, then generate input partial proofs and outputs, return in string format
    """
    partial_proofs = []
    labels = []
    count = 0
    with open(file_path, "r") as f:
        lines = f.readlines()

        for line in lines:
            one_task = json.loads(line)
            sentences, inferences, int_to_all_ancestors_list, relevant_sentences, id_to_int = proof_utils.parse_entailment_step_proof(one_task["proof"], one_task)

            # gather all non leaf nodes
            all_non_leaf_nodes = [infer['rhs'] for infer in inferences]

            # for each non leaf nodes, generate information about (1) what must be included in partial proof, (2) what must NOT included in partial poof, (3) others (chosen randomly)
            info = {}

            # first round, gather descents of each node
            for infer in inferences:
                current_node = infer['rhs']
                info[current_node] = {"must": set(), "avoid": set(), "other": set()}
                for premise in infer['lhs']:
                    if premise in all_non_leaf_nodes:
                        info[current_node]["must"].add(premise)
                        # print(info[premise])
                        info[current_node]["must"] = info[current_node]["must"].union(info[premise]["must"])

            # second round, gather ancester of each node
            for current_node in all_non_leaf_nodes:
                for other_node in all_non_leaf_nodes:
                    if current_node == other_node:
                        info[current_node]["avoid"].add(other_node)
                    elif current_node in info[other_node]['must']:
                        info[current_node]["avoid"].add(other_node)

            # third round, gather all other nodes
            for current_node in all_non_leaf_nodes:
                for other_node in all_non_leaf_nodes:
                    if other_node not in info[current_node]["must"] and other_node not in info[current_node]["avoid"]:
                        info[current_node]["other"].add(other_node)

            for node in info:
                info[node]['must'] = sorted(list(info[node]['must']))
                info[node]['avoid'] = sorted(list(info[node]['avoid']))
                info[node]['must'] = sorted(list(info[node]['must']))
            print("info", info)

            # now for each reasoning step, we generate one partial proof and its corresponding output in string format, which can be directly fed into model for pretraining
            hypothesis_str = f"$hypothesis$ = {one_task['hypothesis']};"
            fact_str = f"$fact$ = {one_task['context']};"

            proof_steps = {}
            for idx, infer in enumerate(inferences):
                current_node = infer["rhs"]
                if current_node == "hypothesis":
                    proof_steps[current_node] = f"{sentences[idx]};"
                else:
                    proof_steps[current_node] = f"{sentences[idx]}: {id_to_int[current_node]};"
            print(proof_steps)

            for current_node in all_non_leaf_nodes:
                label_str = proof_steps[current_node]
                partial_proofs_str = "$partial_proof$ ="
                must_list = sorted(list(info[current_node]['must']))
                other_list = sorted(list(info[current_node]['other']))
                include_set = set()
                for node in must_list:
                   include_set.add(node)
                
                for node in other_list:
                    a = random.choice([0, 1])
                    if a == 1:
                        include_set.add(node)
                        for child_node in info[node]["must"]:
                            # resursively add child node to make partial proof valid
                            include_set.add(child_node)
                
                include_set = sorted(list(include_set))
                for node in include_set:
                    partial_proofs_str += " "
                    partial_proofs_str += proof_steps[node]

                print(must_list, other_list, include_set)
                print("partial_proofs_str:", partial_proofs_str)
                print("label_str:", label_str)

                input_str = f"{hypothesis_str} {fact_str} {partial_proofs_str}"
                partial_proofs.append(input_str)
                labels.append(label_str)


            # break

    print(partial_proofs[502])
    print(labels[502])
    print(len(partial_proofs))
    print(len(labels))
    return partial_proofs, labels

class entailment_bank_dataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.partial_proofs, self.labels = preprocess(file_path)
        assert(len(self.partial_proofs) == len(self.labels))

    def __len__(self):
        return len(self.partial_proofs)
    
    def __getitem__(self, index):
        return self.partial_proofs[index], self.labels[index]

def preprocess_for_verifier(file_path):
    """
        read the data file, generate positive and psudo-negative examples for verifier
    """
    premises_list = []
    conclusion_list = []
    score_list = []
    with open(file_path, "r") as f:
        lines = f.readlines()

        for line in lines:
            one_task = json.loads(line)
            print(one_task)
            sentences, inferences, int_to_all_ancestors_list, relevant_sentences, id_to_int = proof_utils.parse_entailment_step_proof(one_task["proof"], one_task, print_flag=True)

            # first, construct the mapping from sent{x}, int{x} or hypothesis to the corresponding string
            corpus = []
            str_map = {}
            for sent_x, sent_str in one_task["meta"]["triples"].items():
                str_map[sent_x] = sent_str
                corpus.append(sent_str)
            
            for step in inferences:
                int_x = step["rhs"]
                str_map[int_x] = id_to_int[int_x]

            # build corpus for negative examples
            tokenized_corpus = [x.split(" ") for x in corpus]
            bm25 = rank_bm25.BM25Okapi(tokenized_corpus)

            # second, for each proof step, construct valid step and invalid step in string format
            for step in inferences:
                # valid step
                premises = [str_map[premise] for premise in step["lhs"]]
                shuffled_premises = premises.copy()
                random.shuffle(shuffled_premises)
                premises_list.append(". ".join(shuffled_premises))
                conclusion_list.append(str_map[step["rhs"]])
                score_list.append(1.0)

                # invalid step by removing one premise
                deleted_premises = shuffled_premises.copy()
                deleted_premises.pop(random.randint(0, len(deleted_premises) - 1))
                premises_list.append(". ".join(deleted_premises))
                conclusion_list.append(str_map[step["rhs"]])
                score_list.append(0.0)

                # invalid step by changing one premise with a similar one from context using BM25 algorithm
                changed_premises = premises.copy()
                query_idx = random.randint(0, len(changed_premises) - 1)
                query = changed_premises[query_idx]
                tokenized_query = query.split(" ")

                # find the most similar one but not the query itself
                doc_scores = bm25.get_scores(tokenized_query)
                reverse_order = np.argsort(doc_scores)
                similar_sent = corpus[reverse_order[-1]] if corpus[reverse_order[-1]] != query else corpus[reverse_order[-2]]
                assert (similar_sent != query)

                if similar_sent not in set(changed_premises):
                    changed_premises[query_idx] = similar_sent

                    premises_list.append(". ".join(changed_premises))
                    conclusion_list.append(str_map[step["rhs"]])
                    score_list.append(0.0)
                    print("changed premises added")

                # copy one premise as the result
                premises_list.append(". ".join(premises))
                conclusion_list.append(random.choice(premises))
                score_list.append(0.0)

    return premises_list, conclusion_list, score_list
                
class roberta_dataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.premises_list, self.conclusion_list, self.score_list = preprocess_for_verifier(file_path)
        assert (len(self.premises_list) == len(self.score_list))
        assert (len(self.conclusion_list) == len(self.score_list))

    def __len__(self):
        return len(self.score_list)
    
    def __getitem__(self, index):
        return self.premises_list[index], self.conclusion_list[index], self.score_list[index]

if __name__ == "__main__":
    # pass
    _, _, score_list = preprocess_for_verifier(path + "train.jsonl")
    print("score list length", len(score_list))

    _, _, score_list = preprocess_for_verifier(path + "dev.jsonl")
    print("score list length", len(score_list))
    # entailment_bank_dataset(path)