import os
import gradio as gr
from git import Repo
import openai
import time
from datetime import datetime
import json
import re

# Function to clone a GitHub repository
def clone_repo(repo_url, clone_dir="repo_clone", branch=None):
    if os.path.exists(clone_dir):
        os.system(f"rm -rf {clone_dir}")  # Clean up previous clone
    
    if branch:
        Repo.clone_from(repo_url, clone_dir, branch=branch)
        return f"Cloned branch '{branch}' of {repo_url} into {clone_dir}"
    else:
        Repo.clone_from(repo_url, clone_dir)
        return f"Cloned {repo_url} into {clone_dir}"

def runOpenAI(content_batch="", llm_provider="openai", model="gpt-4o", temperature=0, apiKey="", repo_url="", cibles = ""):
        print(f"LLM Provider: {llm_provider}, Model: {model}")
        if llm_provider == "groq":
            client = openai.OpenAI(
                api_key=apiKey,
                base_url="https://api.groq.com/openai/v1"
            )
        else:
            # Clé d'API OpenAI
            openai.api_key = apiKey
            
            client = openai.OpenAI(
                api_key=openai.api_key
            )

        # Prompt détaillé pour analyser le code en français
        prompt = (
            "Vous êtes un assistant spécialisé dans l’analyse de dépôts de code et l’identification des éléments réutilisables et à forte valeur ajoutée. \n"
            "Votre mission est d'identifier de façon exhaustive les snippets de code intéressants du projet fourni. \n"
            f"Sont intéressant les snippets concernant ou mentionnant ces aspects : {cibles}. \n"
            "Pour chaque extrait, construit une fiche markdown bien percutante en markdown :\n"
            f"lien: **Nom du fichier et lien vers le repo et fichier** : Indiquez le fichier où l'extrait se trouve. L'url de base du dépôt cloné est : {repo_url}. Ne garde pas 'repo_clone'\n"
            "code: **Extrait de code** : Fournissez des extraits de code tels que rédigés par l'équipe mais ne le réécrivez pas.\n"
            "description: **Description** : Décrivez ce que fait ce code et son objectif principal.\n"
            "reutilisabilite: **Valeur et Réutilisabilité** : Expliquez pourquoi ce code est précieux ou réutilisable (par exemple : abstraction, modularité, intégration avec des API ou autres services de référence, ou des outils externes).\n"
            "api: **Interactions avec les APIs** : Expliquer les appels d'API, les paramètres utilisés, et le contexte fonctionnel.\n"
            "prompts: **Prompts pour l'IA** : Identifiez les prompts utilisés pour interagir avec des modèles d’IA\n"
            "specificites_fonctionnelles: **Spécificités Fonctionnelles** : Indiquez comment le code répond à des besoins fonctionnels spécifiques du projet.\n"
            + "\n\n".join(content_batch)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    """
                        Vous êtes un expert en analyse de code et en revue technique, spécialisé dans l’identification des composants réutilisables. 
                        Tu ne réponds qu'avec des fiches markdown mise en forme façon carrousel linkedin.
                    """
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]        
        openai_model = model
        start_time = time.time()
        completion = client.chat.completions.create(
            model=openai_model,
            messages=messages,
            temperature=temperature
        )
        # Obtenir la réponse de l'API OpenAI
        openai_response = completion.choices[0].message
        execution_time = time.time() - start_time
        print(f"AI GENERATION OK - {model} - {execution_time} s")
        print(openai_response.content)
        
        return openai_response.content, model, execution_time, temperature, prompt

def get_language_from_extension(filename):
    ext = os.path.splitext(filename)[1].lower()
    lang_map = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".ipynb": "Jupyter Notebook",
        ".sql": "SQL",
        ".java": "Java",
        ".go": "Go",
        ".cs": "C#",
        ".php": "PHP",
        ".rb": "Ruby",
        ".rs": "Rust",
        ".c": "C",
        ".cpp": "C++",
        ".html": "HTML",
        ".css": "CSS"
    }
    return lang_map.get(ext, "Unknown")

def parse_pepite_markdown(markdown_text, chunk_id):
    pepites = []
    # Split by a common delimiter for each pepite, like '### Fiche'
    pepite_blocks = re.split(r'\n###\s+', markdown_text)
    
    for i, block in enumerate(pepite_blocks):
        if not block.strip():
            continue

        # Restore the delimiter for parsing
        block = "### " + block
        
        pepite_data = {}
        pepite_data['pepite_id'] = f"pepite_{chunk_id}_{i}"
        
        title_match = re.search(r'###\s*(.*)', block)
        pepite_data['titre'] = title_match.group(1).strip() if title_match else ""

        lien_match = re.search(r'\*\*Lien\*\*\s*:\s*\[\*\*(.*?)\*\*\]\((.*?)\)', block, re.DOTALL)
        if lien_match:
            pepite_data['fichier_source'] = lien_match.group(1).strip()
            pepite_data['lien'] = lien_match.group(2).strip()
        else: # Fallback for different markdown formats
            lien_match = re.search(r'lien:\s*\*\*.*?\[(.*?)\].*?\((.*?)\)', block, re.DOTALL)
            if lien_match:
                pepite_data['fichier_source'] = lien_match.group(1).strip()
                pepite_data['lien'] = lien_match.group(2).strip()
            else:
                pepite_data['fichier_source'] = ""
                pepite_data['lien'] = ""

        code_match = re.search(r'\*\*Code\*\*\s*:\s*```(.*?)```', block, re.DOTALL)
        pepite_data['code'] = code_match.group(1).strip() if code_match else ""

        description_match = re.search(r'\*\*Description\*\*\s*:\s*(.*?)(?=\n\*\*)', block, re.DOTALL)
        pepite_data['description'] = description_match.group(1).strip() if description_match else ""

        reutilisabilite_match = re.search(r'\*\*Réutilisabilité\*\*\s*:\s*(.*?)(?=\n\*\*)', block, re.DOTALL)
        pepite_data['reutilisabilite'] = reutilisabilite_match.group(1).strip() if reutilisabilite_match else ""

        api_match = re.search(r'\*\*API\*\*\s*:\s*(.*?)(?=\n\*\*)', block, re.DOTALL)
        pepite_data['api'] = api_match.group(1).strip() if api_match else ""

        prompts_match = re.search(r'\*\*Prompts pour l\'IA\*\*\s*:\s*(.*?)(?=\n\*\*)', block, re.DOTALL)
        pepite_data['prompts'] = prompts_match.group(1).strip() if prompts_match else ""

        specificites_match = re.search(r'\*\*Spécificités Fonctionnelles\*\*\s*:\s*(.*)', block, re.DOTALL)
        pepite_data['specificites_fonctionnelles'] = specificites_match.group(1).strip() if specificites_match else ""
        
        pepite_data['langage'] = get_language_from_extension(pepite_data['fichier_source'])

        pepites.append(pepite_data)
        
    return pepites


# Function to extract content and submit to GPT for analysis
def extract_and_process_with_gpt(directory, repo_url, openai_api_key, cibles, llm_provider, model, output_dir):
    files_content = []
    # Collect code snippets from repository
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".py", ".js", ".ts",".ipynb",".sql")):
                filepath = os.path.join(root, file)
                print(filepath)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    files_content.append({"file": filepath, "content": content})

    # Process files in manageable batches
    batch_size = 3
    results = []
    full_pepites_data = []
    summary_pepites_data = []
    total_chunks = 0
    all_results_md = ""

    for i in range(0, len(files_content), batch_size):
        batch = files_content[i : i + batch_size]
        batch_content = [f"File: {item['file']}\nContent:\n{item['content']}" for item in batch]
        pepite_filename = f"pepites_{i}.md"
        pepite_filepath = os.path.join(output_dir, pepite_filename)
        total_chunks += 1

        try:
            print("API Call : ", i)
            gpt_response = runOpenAI(batch_content, repo_url=repo_url, apiKey=openai_api_key, cibles=cibles, llm_provider=llm_provider, model=model)[0]
            results.append(gpt_response)
            with open(pepite_filepath, "w", encoding="utf-8") as fichier:
                fichier.write(gpt_response)
            
            parsed_pepites = parse_pepite_markdown(gpt_response, i)
            full_pepites_data.extend(parsed_pepites)
            
            for pepite in parsed_pepites:
                summary_pepites_data.append({
                    "id": pepite.get('pepite_id'),
                    "titre": pepite.get('titre'),
                    "lien": pepite.get('lien'),
                    "fichier_source": pepite.get('fichier_source'),
                    "langage": pepite.get('langage')
                })

            all_results_md += gpt_response + "\n\n---\n\n"
            yield all_results_md
        except Exception as e:
            error_message = f"Error processing batch: {str(e)}"
            results.append(error_message)
            with open(pepite_filepath, "w", encoding="utf-8") as f:
                f.write(error_message)
            all_results_md += error_message + "\n\n---\n\n"
            yield all_results_md

    # Create full index file
    full_index_path = os.path.join(output_dir, "_index_full.json")
    with open(full_index_path, "w", encoding="utf-8") as f:
        json.dump(full_pepites_data, f, indent=4, ensure_ascii=False)

    # Create summary index file
    summary_index_path = os.path.join(output_dir, "_index_summary.json")
    with open(summary_index_path, "w", encoding="utf-8") as f:
        json.dump(summary_pepites_data, f, indent=4, ensure_ascii=False)

    # Final concatenated results
    final_pepites = "\n\n".join(results)
    main_pepites_file = os.path.join(output_dir, "all_pepites.md")
    with open(main_pepites_file, "w", encoding="utf-8") as fichier:
        fichier.write(final_pepites)


# Gradio Interface
def process_repo(repo_url, openai_api_key, cibles, llm_provider, model):
    try:
        # Parse the input URL to handle subdirectories
        url_pattern = r"https://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+)(?:/(.*))?)?"
        match = re.match(url_pattern, repo_url)
        
        if not match:
            yield "URL de dépôt invalide. Veuillez fournir une URL GitHub valide."
            return

        owner, repo = match.group(1), match.group(2)
        branch = match.group(3)
        subdirectory = match.group(4) or ""

        clone_url = f"https://github.com/{owner}/{repo}.git"
        repo_name = repo
        
        # Define the base URL for links, including the branch
        link_base_url = f"https://github.com/{owner}/{repo}/tree/{branch}" if branch else f"https://github.com/{owner}/{repo}/tree/main"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{repo_name}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        clone_dir = os.path.join(output_dir, "repo_clone")
        
        yield f"Cloning {clone_url} (branch: {branch or 'default'})..."
        clone_repo(clone_url, clone_dir, branch=branch)
        
        analysis_path = os.path.join(clone_dir, subdirectory)
        if not os.path.isdir(analysis_path):
            yield f"Error: Subdirectory '{subdirectory}' not found in the repository."
            return

        yield f"Repository cloned. Starting analysis of '{subdirectory or 'root'}' in {output_dir}..."
        
        # The function now yields updates for the Gradio interface
        for pepites_update in extract_and_process_with_gpt(analysis_path, link_base_url, openai_api_key, cibles, llm_provider, model, output_dir):
            yield pepites_update

        yield f"Analyse terminée. Les résultats sont dans le dossier : {output_dir}"
    except Exception as e:
        yield f"Error processing repository: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# Pépites du Hackathon")
    gr.Markdown("## Sélectionnez un repo github pour en extraire des pépites selon vos centres d'intérêts")

    def update_models(provider, apiKey):
        if not apiKey:
            return gr.Dropdown(choices=[], value=None, interactive=False)
        try:
            if provider == "groq":
                client = openai.OpenAI(
                    api_key=apiKey,
                    base_url="https://api.groq.com/openai/v1"
                )
            else:
                client = openai.OpenAI(api_key=apiKey)
            
            models_list = client.models.list()
            model_ids = sorted([model.id for model in models_list.data])
            
            return gr.Dropdown(choices=model_ids, value=model_ids[0] if model_ids else None, interactive=True)
        except Exception as e:
            print(f"Error fetching models: {e}")
            return gr.Dropdown(choices=[], value=f"Erreur: {e}", interactive=False)

    with gr.Row():
        openai_api_key = gr.Textbox(
            label="Clé API",
            placeholder="Entrez votre clé magique 🙂",
            type="password"
        )

    with gr.Row():
        llm_provider_input = gr.Dropdown(
            label="Fournisseur LLM",
            choices=["openai", "groq"],
            value="openai"
        )
        model_input = gr.Dropdown(
            label="Modèle",
            choices=[],
            interactive=False
        )
    
    llm_provider_input.change(fn=update_models, inputs=[llm_provider_input, openai_api_key], outputs=model_input)
    openai_api_key.change(fn=update_models, inputs=[llm_provider_input, openai_api_key], outputs=model_input)
    with gr.Row():
        repo_url_input = gr.Textbox(
            label="URL du repo Github",
            placeholder="Entrez l'url du dépôt",
        )
    with gr.Row():
        cibles_input = gr.Textbox(
            label="Que ciblez vous ?",
            placeholder="Précisez vos cibles : appels d'API, xpaths, design patterns",
        )
    analyze_button = gr.Button("Fouiller")
    with gr.Row():
        output_box = gr.Markdown()
    
    analyze_button.click(process_repo, inputs=[repo_url_input, openai_api_key, cibles_input, llm_provider_input, model_input], outputs=[output_box])



if __name__ == "__main__":
    demo.launch()
