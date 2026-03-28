import argparse
import json
import os
import re
from os import getenv

from datasets import Dataset
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


def load_benchmark_dataset(dataset_path):
	resolved_path = dataset_path
	if not os.path.isabs(resolved_path):
		repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		resolved_path = os.path.join(repo_root, resolved_path)

	if os.path.isfile(resolved_path) and resolved_path.endswith(".json"):
		with open(resolved_path, "r", encoding="utf-8") as f:
			payload = json.load(f)

		if isinstance(payload, dict):
			test_rows = payload.get("test", [])
			validation_rows = payload.get("validation", [])
		elif isinstance(payload, list):
			test_rows = payload
			validation_rows = []
		else:
			raise ValueError(
				f"Unsupported JSON dataset structure in {resolved_path}. "
				"Expected a list or a dict with test/validation keys."
			)

		test_dataset = Dataset.from_list(test_rows)
		validation_dataset = Dataset.from_list(validation_rows)
		return test_dataset, validation_dataset

	dataset = load_dataset(dataset_path)
	test_dataset = dataset["test"]
	validation_dataset = dataset["validation"]
	return test_dataset, validation_dataset


def load_client(args):
	api_key = args.api_key or getenv("OPENROUTER_API_KEY")
	if not api_key:
		raise ValueError(
			"Missing API key. Set OPENROUTER_API_KEY or pass --api_key."
		)

	client = OpenAI(
		base_url=args.base_url,
		api_key=api_key,
	)
	return client


def build_prompt(*, prompt_path, question, choices, category="", label=""):
	with open(prompt_path, "r", encoding="utf-8") as f:
		prompt_template = f.read()
	prompt = prompt_template.replace("{question}", question)
	for i, choice in enumerate(choices):
		prompt = prompt.replace(f"{{choice_{i}}}", choice)
	prompt = prompt.replace("{label}", label)
	if "{$}" in prompt:
		prompt = prompt.replace("{$}", category)
	return prompt


def resolve_prompt_path(use_cot):
	if use_cot:
		return os.path.join("prompt_lib", "openai_cot.txt")
	return os.path.join("prompt_lib", "base.txt")


def sanitize_model_name(model_name):
	sanitized = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")
	return sanitized.lower()


def resolve_output_path(args):
	if args.output_file:
		filename = args.output_file
	else:
		prompt_mode = "cot" if args.use_cot else "base"
		model_name = sanitize_model_name(args.model)
		filename = f"{model_name}_{prompt_mode}_{args.n_shots}shot.json"
		if args.start_index > 0:
			filename = f"{model_name}_{prompt_mode}_{args.n_shots}shot_start{args.start_index}.json"
	return os.path.join(args.output_dir, filename)


def generate_model_output(client, args, prompt):
	completion = client.chat.completions.create(
		model=args.model,
		temperature=0,
		max_tokens=args.max_new_tokens,
		messages=[{"role": "user", "content": prompt}],
		extra_headers={
			"HTTP-Referer": args.referer,
			"X-OpenRouter-Title": args.title,
		},
	)
	return completion.choices[0].message.content or ""


def build_domain_shots(validation_dataset):
	domain_shots = {}
	for row in validation_dataset:
		domain = row["domain"]
		domain_shots.setdefault(domain, []).append(row)
	return domain_shots


def build_nshot_prefix(*, nshot_prompt_path, domain_shots, n_shots, domain):
	if n_shots <= 0:
		return ""

	shots = domain_shots.get(domain, [])
	if len(shots) < n_shots:
		raise ValueError(
			f"Not enough validation shots for domain '{domain}': "
			f"required {n_shots}, found {len(shots)}."
		)

	shot_prompts = []
	for shot in shots[:n_shots]:
		shot_prompts.append(
			build_prompt(
				prompt_path=nshot_prompt_path,
				question=shot["query"],
				choices=shot["choices"],
				category=shot["domain"],
				label=shot["answer"],
			)
		)

	return "\n\n".join(shot_prompts)


def run_benchmark(client, test_dataset, validation_dataset, output_path, args):
	prompt_path = resolve_prompt_path(args.use_cot)
	nshot_prompt_path = os.path.join("prompt_lib", "nshot.txt")
	domain_shots = build_domain_shots(validation_dataset)

	output_data = []
	assert not os.path.exists(output_path), f"Output file {output_path} already exists. Please remove it before running the benchmark to avoid overwriting results."

	if args.n_shots > 0 and len(validation_dataset) == 0:
		raise ValueError(
			"n-shot prompting requires a validation split. "
			"No validation rows were found in the dataset."
		)

	test_dataset = test_dataset.select(range(args.start_index, len(test_dataset)))
	for row in tqdm(test_dataset, total=len(test_dataset)):
		base_prompt = build_prompt(
			prompt_path=prompt_path,
			question=row["query"],
			choices=row["choices"],
			category=row["domain"],
		)

		nshot_prefix = build_nshot_prefix(
			nshot_prompt_path=nshot_prompt_path,
			domain_shots=domain_shots,
			n_shots=args.n_shots,
			domain=row["domain"],
		)
		prompt = base_prompt if nshot_prefix == "" else f"{nshot_prefix}\n\n{base_prompt}"

		reply = generate_model_output(client, args, prompt)
		if reply == "":
			print(f"Received empty reply for question: '{row['query']}'. Halting this test!.")
			break

		output_data.append(
			{
				"query": row["query"],
				"choices": row["choices"],
				"domain": row["domain"],
				"answer" : row["answer"],
				"model_output": reply,
			}
		)

		with open(output_path, "w", encoding="utf-8") as f:
			json.dump(output_data, f, indent=4)

	print(f"Saved {len(output_data)} benchmark results to {output_path}")


def main(args):
	if args.n_shots < 0:
		raise ValueError("n_shots must be >= 0.")

	test_dataset, validation_dataset = load_benchmark_dataset(args.dataset_path)
	client = load_client(args)
	output_path = resolve_output_path(args)
	run_benchmark(client, test_dataset, validation_dataset, output_path, args)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dataset_path",
		type=str,
		default="./data/sake-dataset-sample.json",
		help="Dataset name/path. Uses test split for scoring and validation split for few-shot examples.",
	)
	parser.add_argument(
		"--n_shots",
		type=int,
		default=0,
		help="Number of domain-matched validation examples to prepend to each test prompt.",
	)
	parser.add_argument(
		"--model",
		type=str,
		default="stepfun/step-3.5-flash:free",
		help="OpenRouter model name.",
	)
	parser.add_argument(
		"--base_url",
		type=str,
		default="https://openrouter.ai/api/v1",
		help="OpenAI-compatible endpoint base URL.",
	)
	parser.add_argument(
		"--api_key",
		type=str,
		default=None,
		help="OpenRouter API key. If omitted, uses OPENROUTER_API_KEY.",
	)
	parser.add_argument(
		"--max_new_tokens",
		type=int,
		default=16,
		help="Maximum new tokens to generate for each question.",
	)
	parser.add_argument(
		"--use_cot",
		action="store_true",
		help="Enable a reasoning-style prompt. Disabled by default.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="./outputs",
		help="Directory to save the evaluation outputs.",
	)
	parser.add_argument(
		"--output_file",
		type=str,
		default=None,
		help="Optional output JSON filename. Defaults to <model_name>_<base|cot>_<n>shot.json.",
	)
	parser.add_argument(
		"--referer",
		type=str,
		default="http://localhost",
		help="Optional HTTP-Referer header for OpenRouter ranking.",
	)
	parser.add_argument(
		"--title",
		type=str,
		default="sake-benchmark",
		help="Optional X-Title header for OpenRouter ranking.",
	)
	parser.add_argument(
		"--start_index",
		type=int,
		default=0,
		help="Starting index in the dataset to allow resuming from a specific point.",
	)
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)
	main(args)
