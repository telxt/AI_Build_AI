import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
import evaluate
import json
import argparse
import numpy as np
from tqdm import tqdm

def compute_log_likelihood_batch(prompts, continuations, tokenizer, model, device):
    input_ids = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors='pt').input_ids
    continuation_ids = tokenizer(continuations, add_special_tokens=False, padding=True, return_tensors='pt').input_ids

    full_input_ids = torch.cat([input_ids, continuation_ids], dim=-1).to(device)
    labels = torch.full_like(full_input_ids, -100)
    labels[:, input_ids.shape[1]:] = continuation_ids

    with torch.no_grad():
        outputs = model(full_input_ids, labels=labels)
        log_likelihoods = -outputs.loss.cpu().numpy() * (labels != -100).sum(-1).cpu().numpy()
    return log_likelihoods

def eval(device, batch_size, tasks, model_name, output_file):
    device = device
    batch_size = batch_size
    tasks = tasks
    # Mapping from task to possible labels
    task_to_labels = json.loads('task2labels.json')

    # Mapping from task to evaluation metric
    task_to_metric = json.loads('task2metric.json')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    results = {}

    for task in tasks:
        print(f"\nEvaluating {task}...")
        if task in task_to_labels:
            dataset = datasets.load_dataset('glue', task)
            data = dataset["validation_matched" if task == "mnli" else "validation"]

            if task == "sst2" or task == "cola":
                texts = data['sentence']
            elif task in ["mrpc", "rte", "wnli"]:
                texts = list(zip(data['sentence1'], data['sentence2']))
            elif task == "qqp":
                texts = list(zip(data['question1'], data['question2']))
            elif task == "stsb":
                texts = list(zip(data['sentence1'], data['sentence2']))
            elif task == "mnli":
                texts = list(zip(data['premise'], data['hypothesis']))
            elif task == "qnli":
                texts = list(zip(data['question'], data['sentence']))
            labels = data['label']
            possible_labels = task_to_labels[task]

            predictions = []
            for batch_start in tqdm(range(0, len(texts), batch_size)):
                batch_end = min(batch_start + batch_size, len(texts))
                batch_texts = texts[batch_start:batch_end]
                batch_prompts = []
                if task == "mnli":
                    for text1, text2 in batch_texts:
                        batch_prompts.append(f"Premise: {text1}\nHypothesis: {text2}\nRelationship:")
                elif task == "qnli":
                    for text1, text2 in batch_texts:
                        batch_prompts.append(f"Question: {text1}\nSentence: {text2}\nEntailment:")
                elif task == "qqp":
                    for text1, text2 in batch_texts:
                        batch_prompts.append(f"Question1: {text1}\nQuestion2: {text2}\nDuplicate:")
                elif task == "rte" or task == "wnli":
                    for text1, text2 in batch_texts:
                        batch_prompts.append(f"Premise: {text1}\nHypothesis: {text2}\nEntailment:")
                elif task == "mrpc":
                    for text1, text2 in batch_texts:
                        batch_prompts.append(f"Sentence1: {text1}\nSentence2: {text2}\nEquivalence:")
                elif task == "cola" or task == "sst2":
                    for text in batch_texts:
                        batch_prompts.append(f"Sentence: {text}\nAcceptability:" if task == "cola" else f"Review: {text}\nSentiment:")

                label_scores = []
                for label in possible_labels:
                    label_continuations = [f" {label}"] * len(batch_prompts)
                    label_scores.append(compute_log_likelihood_batch(batch_prompts, label_continuations))

                pred_indices = np.argmax(label_scores, axis=0)
                predictions.extend(pred_indices)

            metric = evaluate.load(task_to_metric[task])
            result = metric.compute(predictions=predictions, references=labels)
            results[task] = {task_to_metric[task]: result[task_to_metric[task]]}
            print(f"Results for {task}: {results[task]}")

        elif task in ["arc_easy", "arc_challenge", "mmlu"]:
            if task in ["arc_easy", "arc_challenge"]:
                dataset = datasets.load_dataset("ai2_arc", 'ARC-Easy' if task == "arc_easy" else 'ARC-Challenge')
                data = dataset['validation']
            elif task == "mmlu":
                dataset = datasets.load_dataset("cais/mmlu", "all") 
                data = dataset["validation"]
            else:
                raise ValueError(f"Task {task} not supported")

            texts = []
            labels = []
            for item in data:
                question = item['question']
                if task in ["arc_easy", "arc_challenge"]:
                    choices = item['choices']['text']
                    answer = item['answerKey']
                    labels.append(ord(answer) - ord('A'))
                elif task == "mmlu":
                    choices = item['choices']
                    answer = item['answer']
                    labels.append(answer)
                texts.append({
                    'question': question,
                    'choices': choices
                })

            predictions = []
            for batch_start in tqdm(range(0, len(texts), batch_size)):
                batch_end = min(batch_start + batch_size, len(texts))
                batch = texts[batch_start:batch_end]

                batch_prompts = [f"Question: {example['question']}\nAnswer:" for example in batch]
                max_num_choices = max(len(example['choices']) for example in batch)

                choice_scores = np.full((len(batch), max_num_choices), -np.inf, dtype=np.float32)

                for choice_idx in range(max_num_choices):
                    choice_continuations = []
                    valid_indices = []
                    for i, example in enumerate(batch):
                        if choice_idx < len(example['choices']):
                            choice = example['choices'][choice_idx]
                            choice_continuations.append(f" {choice}")
                            valid_indices.append(i)
                        else:
                            choice_scores[i, choice_idx] = -np.inf
                            choice_continuations.append("")

                    if len(choice_continuations) > 0:
                        log_likelihoods = compute_log_likelihood_batch(
                            [batch_prompts[i] for i in valid_indices],
                            [choice_continuations[i] for i in range(len(choice_continuations)) if batch[i]['choices'] and choice_idx < len(batch[i]['choices'])]
                        )
                        for idx, ll in zip(valid_indices, log_likelihoods):
                            choice_scores[idx, choice_idx] = ll

                pred_indices = np.argmax(choice_scores, axis=1)
                predictions.extend(pred_indices)

            metric = evaluate.load("accuracy")
            result = metric.compute(predictions=predictions, references=labels)
            results[task] = {"accuracy": result["accuracy"]}
            print(f"Results for {task}: {results[task]}")
            
        elif task == "stsb":
            dataset = datasets.load_dataset('glue', 'stsb')
            data = dataset['validation']

            texts = list(zip(data['sentence1'], data['sentence2']))
            labels = data['label'] 

            predictions = []
            for batch_start in tqdm(range(0, len(texts), batch_size)):
                batch_end = min(batch_start + batch_size, len(texts))
                batch_texts = texts[batch_start:batch_end]
                batch_prompts = [f"Sentence1: {text1}\nSentence2: {text2}\nSimilarity score:" for text1, text2 in batch_texts]

                possible_scores = [f" {score}" for score in np.arange(0, 5.5, 0.5).tolist()]
                label_scores = []
                for score in possible_scores:
                    log_likelihood = compute_log_likelihood_batch(batch_prompts, [score]*len(batch_prompts))
                    label_scores.append(log_likelihood)

                label_scores = np.array(label_scores)  # Shape: (num_scores, batch_size)
                pred_indices = np.argmax(label_scores, axis=0)
                predicted_scores = [float(score) for score in np.arange(0, 5.5, 0.5).tolist()]
                batch_predictions = [predicted_scores[idx] for idx in pred_indices]
                predictions.extend(batch_predictions)

            pearson = evaluate.load("pearsonr")
            spearman = evaluate.load("spearmanr")
            pearson_result = pearson.compute(predictions=predictions, references=labels)
            spearman_result = spearman.compute(predictions=predictions, references=labels)
            results[task] = {"pearson": pearson_result["pearsonr"], "spearmanr": spearman_result["spearmanr"]}
            print(f"Results for {task}: {results[task]}")
        elif task == "squad":
            dataset = datasets.load_dataset('squad')
            data = dataset['validation']

            texts = list(zip(data['context'], data['question']))
            labels = list(d['answers']['text'] for d in data)

            predictions = []
            for batch_start in tqdm(range(0, len(texts), batch_size)):
                batch_end = min(batch_start + batch_size, len(texts))
                batch_texts = texts[batch_start:batch_end]
                batch_prompts = [f"Context: {context}\nQuestion: {question}\nAnswer:" for question, context in batch_texts]

                label_scores = []
                for label in labels[batch_start:batch_end]:
                    log_likelihood = compute_log_likelihood_batch(batch_prompts, [f" {label}"]*len(batch_prompts))
                    label_scores.append(log_likelihood)

                pred_indices = np.argmax(label_scores, axis=0)
                predictions.extend(pred_indices)

            metric = evaluate.load("f1")
            result = metric.compute(predictions=predictions, references=labels)
            results[task] = {"f1": result["f1"]}
            print(f"Results for {task}: {results[task]}")
            with open(output_file, "w") as f:
                json.dump(results, f)
     
def pipe(args):
    tasks = args.tasks.split(" ")
    models = args.models.split(" ")
    batch_size = args.batch_size
    device = args.device
    output_files = args.output_files
    for model, output_file in zip(models, output_files):
        eval(device, batch_size, tasks, model, output_file)
    
    