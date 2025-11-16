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
            "Pour chaque extrait, construit une fiche JSON.\n"
            f"L'url de base du dépôt cloné est : {repo_url}. Ne garde pas 'repo_clone' dans le lien.\n"
            + "\n\n".join(content_batch)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    """
                        Vous êtes un expert en analyse de code. Votre mission est de produire un JSON contenant une liste de "pépites" (extraits de code).
                        Pour chaque pépite, vous devez identifier à quelles cibles de l'utilisateur elle correspond.
                        Le schéma de sortie doit être le suivant :
                        {
                          "pepites": [
                            {
                              "titre": "Titre descriptif de la pépite",
                              "lien": "URL complète vers le fichier sur GitHub",
                              "fichier_source": "Chemin relatif du fichier dans le dépôt",
                              "code": "L'extrait de code pertinent",
                              "description": "Description de ce que fait le code et son objectif",
                              "reutilisabilite": "Explication de la valeur et de la réutilisabilité du code",
                              "api": "Description des interactions avec des APIs, si applicable",
                              "prompts": "Identification des prompts pour l'IA, si applicable",
                              "specificites_fonctionnelles": "Comment le code répond à des besoins fonctionnels spécifiques",
                              "cibles_concernees": ["cible_1", "cible_2"]
                            }
                          ]
                        }
                        Le champ "cibles_concernees" doit être un tableau listant les cibles pertinentes pour la pépite.
                        Vous ne devez répondre qu'avec le JSON, sans aucun commentaire ou texte supplémentaire.
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
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        # Obtenir la réponse de l'API OpenAI
        openai_response = completion.choices[0].message
        execution_time = time.time() - start_time
        print(f"AI GENERATION OK - {model} - {execution_time} s")
        
        # Charger la réponse JSON
        try:
            response_json = json.loads(openai_response.content)
            print(response_json)
            return response_json, messages
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from OpenAI response")
            print(openai_response.content)
            return {"pepites": []}, messages

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

def runOpenAISummary(readme_content="", llm_provider="openai", model="gpt-4o", temperature=0, apiKey=""):
    print(f"LLM Provider for Summary: {llm_provider}, Model: {model}")
    if llm_provider == "groq":
        client = openai.OpenAI(
            api_key=apiKey,
            base_url="https://api.groq.com/openai/v1"
        )
    else:
        client = openai.OpenAI(api_key=apiKey)

    messages = [
        {
            "role": "system",
            "content": (
                """
                    Vous êtes un expert en analyse de documentation de projet. Votre mission est de produire un résumé général concis et pertinent à partir du contenu d'un fichier README.md.
                    Le résumé doit être en français et doit capturer l'essence du projet : son objectif, ses fonctionnalités principales et les technologies utilisées.
                    Vous devez répondre uniquement avec un objet JSON contenant une seule clé "resume_projet".
                    Exemple de sortie :
                    {
                      "resume_projet": "Ce projet est une application web pour la gestion de tâches, développée en React et Node.js. Elle permet aux utilisateurs de créer, suivre et organiser leurs tâches quotidiennes."
                    }
                """
            ),
        },
        {
            "role": "user",
            "content": f"Voici le contenu du README.md, génère un résumé du projet :\n\n{readme_content}",
        },
    ]

    start_time = time.time()
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"}
    )
    execution_time = time.time() - start_time
    print(f"AI SUMMARY GENERATION OK - {model} - {execution_time} s")

    try:
        response_json = json.loads(completion.choices[0].message.content)
        print(response_json)
        return response_json, messages
    except (json.JSONDecodeError, KeyError):
        print("Error: Failed to decode JSON or key not found in OpenAI summary response")
        print(completion.choices[0].message.content)
        return {"resume_projet": "Impossible de générer le résumé du projet."}, messages


# Function to extract content and submit to GPT for analysis
def extract_and_process_with_gpt(directory, repo_url, openai_api_key, cibles, llm_provider, model, output_dir):
    all_prompts = []
    
    # 1. Traitement du README pour le résumé
    project_summary = ""
    readme_path = os.path.join(directory, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        summary_response, summary_prompt = runOpenAISummary(readme_content, llm_provider=llm_provider, model=model, apiKey=openai_api_key)
        project_summary = summary_response.get("resume_projet", "Résumé non disponible.")
        all_prompts.append({"type": "resume", "prompt": summary_prompt})
        yield f"Résumé du projet généré.\n"
    else:
        yield "Aucun fichier README.md trouvé. Le résumé du projet ne sera pas généré.\n"

    # 2. Collecte des fichiers de code
    files_content = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".py", ".js", ".ts", ".ipynb", ".sql")):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        files_content.append({"file": filepath, "content": content})
                except Exception as e:
                    print(f"Could not read file {filepath}: {e}")

    # 3. Traitement par lots pour les pépites
    batch_size = 3
    all_pepites = []
    total_files = len(files_content)
    processed_files = 0

    for i in range(0, total_files, batch_size):
        batch = files_content[i : i + batch_size]
        batch_content = [f"File: {item['file']}\nContent:\n{item['content']}" for item in batch]
        
        try:
            response_json, pepite_prompt = runOpenAI(batch_content, repo_url=repo_url, apiKey=openai_api_key, cibles=cibles, llm_provider=llm_provider, model=model)
            pepites = response_json.get("pepites", [])
            all_prompts.append({"type": f"pepites_batch_{i//batch_size}", "prompt": pepite_prompt})
            
            for pepite in pepites:
                pepite['langage'] = get_language_from_extension(pepite.get('fichier_source', ''))
            
            all_pepites.extend(pepites)
            processed_files += len(batch)
            yield f"Analyse en cours... {processed_files}/{total_files} fichiers traités."

        except Exception as e:
            error_message = f"Erreur lors du traitement d'un lot : {str(e)}"
            print(error_message)
            yield error_message

    # 4. Création du fichier JSON final
    final_output = {
        "resume_projet": project_summary,
        "pepites": all_pepites,
        "prompts_utilises": all_prompts
    }
    
    output_file_path = os.path.join(output_dir, "pepites_project.json")
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    yield f"Analyse terminée. Le fichier de résultats a été sauvegardé ici : {output_file_path}"


# Gradio Interface
def process_repo(repo_url, openai_api_key, cibles, llm_provider, model):
    log_message = ""
    summary_message = ""
    stats_message = ""
    prompts_json = []
    
    def yield_updates():
        return {
            output_box: log_message,
            summary_output: summary_message,
            stats_output: stats_message,
            prompts_output: prompts_json
        }

    try:
        url_pattern = r"https://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+)(?:/(.*))?)?"
        match = re.match(url_pattern, repo_url)
        
        if not match:
            log_message = "URL de dépôt invalide."
            yield yield_updates()
            return

        owner, repo, branch, subdirectory = match.groups()
        subdirectory = subdirectory or ""
        clone_url = f"https://github.com/{owner}/{repo}.git"
        repo_name = repo
        link_base_url = f"https://github.com/{owner}/{repo}/tree/{branch or 'main'}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Créer un dossier de base 'data' pour toutes les sorties
        base_output_dir = "data"
        output_dir = os.path.join(base_output_dir, f"{repo_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        clone_dir = os.path.join(output_dir, "repo_clone")
        
        log_message += f"Clonage de {clone_url}...\n"
        yield yield_updates()
        clone_repo(clone_url, clone_dir, branch=branch)
        
        analysis_path = os.path.join(clone_dir, subdirectory)
        if not os.path.isdir(analysis_path):
            log_message += f"Erreur: Le sous-dossier '{subdirectory}' n'a pas été trouvé.\n"
            yield yield_updates()
            return

        log_message += "Dépôt cloné. Début de l'analyse...\n"
        yield yield_updates()
        
        final_json_path = ""
        for update in extract_and_process_with_gpt(analysis_path, link_base_url, openai_api_key, cibles, llm_provider, model, output_dir):
            log_message += update + "\n"
            if "sauvegardé ici" in update:
                final_json_path = update.split("sauvegardé ici : ")[-1].strip()
            yield yield_updates()

        if final_json_path and os.path.exists(final_json_path):
            with open(final_json_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            
            summary_message = results.get("resume_projet", "Aucun résumé.")
            pepites = results.get("pepites", [])
            total_pepites = len(pepites)
            stats_message = f"### **Nombre total de pépites : {total_pepites}**\n"
            
            if total_pepites > 0:
                lang_counts = {}
                for pepite in pepites:
                    lang = pepite.get("langage", "Inconnu")
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                
                stats_message += "\n**Répartition par langage :**\n"
                for lang, count in sorted(lang_counts.items()):
                    stats_message += f"- **{lang}** : {count}\n"
            
            prompts_json = results.get("prompts_utilises", [])
        
        log_message += f"\nAnalyse terminée. Fichier de résultats : {final_json_path}\n"
        yield yield_updates()

    except Exception as e:
        log_message += f"\nErreur inattendue : {str(e)}\n"
        yield yield_updates()

with gr.Blocks() as demo:
    gr.Markdown("# Pépites du Hackathon")
    gr.Markdown("## Analysez un dépôt GitHub pour en extraire des pépites de code.")

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
        openai_api_key = gr.Textbox(label="Clé API", placeholder="Entrez votre clé OpenAI ou Groq", type="password")

    with gr.Row():
        llm_provider_input = gr.Dropdown(label="Fournisseur LLM", choices=["openai", "groq"], value="openai")
        model_input = gr.Dropdown(label="Modèle", choices=[], interactive=False)
    
    llm_provider_input.change(fn=update_models, inputs=[llm_provider_input, openai_api_key], outputs=model_input)
    openai_api_key.change(fn=update_models, inputs=[llm_provider_input, openai_api_key], outputs=model_input)
    
    repo_url_input = gr.Textbox(label="URL du repo Github", placeholder="Entrez l'URL du dépôt")
    cibles_input = gr.Textbox(label="Que ciblez-vous ?", placeholder="Ex: appels d'API, design patterns, etc.")
    
    analyze_button = gr.Button("Analyser le Dépôt")
    
    output_box = gr.Textbox(label="Log d'analyse", lines=10, interactive=False)
    summary_output = gr.Textbox(label="Résumé du Projet", lines=5, interactive=False)
    stats_output = gr.Markdown(label="Statistiques des Pépites")
    
    with gr.Accordion("Voir les prompts utilisés", open=False):
        prompts_output = gr.JSON(label="Prompts envoyés à l'API")

    analyze_button.click(
        process_repo, 
        inputs=[repo_url_input, openai_api_key, cibles_input, llm_provider_input, model_input], 
        outputs=[output_box, summary_output, stats_output, prompts_output]
    )



if __name__ == "__main__":
    demo.launch()
