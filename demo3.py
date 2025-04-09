import os
import json
import yaml
import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import queue
import sys
from io import StringIO
import time

# Import all necessary modules from your original code
import alfworld
import alfworld.agents.environment
from oracle import oracle_support
from user_sim import LLMUserAgent
from openai import OpenAI, AzureOpenAI

# Custom stdout to capture print statements
class StdoutRedirector(StringIO):
    def __init__(self, text_widget, queue):
        super().__init__()
        self.text_widget = text_widget
        self.queue = queue
        
    def write(self, string):
        self.queue.put(('log', string))
        
    def flush(self):
        pass

# Helper functions from original code
def load_yaml(file):
    with open(file) as f:
        return yaml.safe_load(f)

def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)
        
def load_pddl(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def load_txt(file):
    with open(file, 'r') as f:
        return f.read()
        
def generate_human_task_description(task_desc):
    """Generate a more natural human description of the task using LLM"""
    prompt = [
        {"role": "system", "content": f"You are a helpful assistant that converts formal task descriptions into natural human instructions. Keep your response concise and conversational. Convert this formal task description into a natural instruction as if a human is explaining the task to an AI assistant: '{task_desc}'. Start with 'Your task is to' and make it sound conversational but brief, If there is an object mentioned ask the AI to first find the object in the environment. Don't use bullet points or numbered lists, just a simple statement. For example, 'Your task is to look at pencil under the desklamp.' This can be converted to 'Your task is to look at pencil under the desklamp. First, Find the pencil, then locate the desklamp and then examine it under the lamp.'"}
    ]
    
    try:
        response = llm(prompt, stop=[])
        return response
    except Exception as e:
        print(f"Error generating task description: {str(e)}")
        # Fallback if LLM fails
        return f"Your task is to {task_desc.lower().rstrip('.')}. Please help me with this."

def get_openai_client(use_azure=False):
    if use_azure:
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2023-03-15-preview"
        )
    else:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
    return client

def llm(prompt, stop=["\n"], use_azure=False):
    client = get_openai_client(use_azure)
    
    if use_azure:
        model = "gpt4o-mini" 
    else:
        model = "gpt-4o"
    
    response = client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop
    )
    content = response.choices[0].message.content
    
    # Post-process to ensure we stop at the first occurrence of any stop sequence
    for stop_seq in stop:
        index = content.find(stop_seq)
        if index != -1:
            content = content[:index]
    
    return content.strip()

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

# Function to scan for task folders
def get_task_folders(configs_root_dir="task_configs"):
    """Scan the configs directory to find all task subfolders"""
    if not os.path.exists(configs_root_dir):
        # If the directory doesn't exist, return a default set for example purposes
        return ["pick_and_place", "pick_clean_then_place", "pick_heat_then_place", 
                "pick_cool_then_place", "look_at_obj", "pick_two_obj"]
    
    # Get all subdirectories in the configs root
    task_folders = [d for d in os.listdir(configs_root_dir) 
                   if os.path.isdir(os.path.join(configs_root_dir, d))]
    
    # Filter to only include folders that contain both config files
    valid_task_folders = []
    for folder in task_folders:
        folder_path = os.path.join(configs_root_dir, folder)
        if (os.path.exists(os.path.join(folder_path, "base_config.yaml")) and
            os.path.exists(os.path.join(folder_path, "base_config2.yaml"))):
            valid_task_folders.append(folder)
    
    return valid_task_folders

# Main GUI Application
class AlfworldGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ReSpAct AlfWorld Agent Interface")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f5f5f5")
        
        self.message_queue = queue.Queue()
        self.human_input_queue = queue.Queue()
        self.waiting_for_human_input = False
        
        # Config settings
        self.configs_root_dir = "task_configs"  # Default directory for task configs
        self.current_task = tk.StringVar()
        
        self.setup_ui()
        self.setup_styles()
        
        # Start the queue processing
        self.root.after(100, self.process_queue)
        
        # Load available task folders
        self.load_task_folders()
        
    def load_task_folders(self):
        """Load available task folders into the task dropdown"""
        task_folders = get_task_folders(self.configs_root_dir)
        
        # If we have valid tasks, set the first one as default
        if task_folders:
            self.task_dropdown['values'] = task_folders
            self.current_task.set(task_folders[0])
        else:
            self.message_queue.put(('system', f"No valid task configurations found in {self.configs_root_dir}. Please check directory structure."))
            self.task_dropdown['values'] = ["No tasks found"]
            self.current_task.set("No tasks found")
        
    def setup_styles(self):
        # Font sizes - adjust these values to change font size throughout the app
        self.title_font_size = 14
        self.main_font_size = 11  
        self.system_font_size = 10  
        self.reward_font_size = 13  
        
        # Configure styles for different message types in the main frame
        self.main_frame.tag_configure("think", foreground="#6a0dad", font=("Helvetica", self.main_font_size, "italic"))
        self.main_frame.tag_configure("act", foreground="#008000", font=("Helvetica", self.main_font_size, "bold"))
        self.main_frame.tag_configure("speak", foreground="#0000CD", font=("Helvetica", self.main_font_size, "bold"))
        self.main_frame.tag_configure("observation", foreground="#B8860B", font=("Helvetica", self.main_font_size))
        self.main_frame.tag_configure("human", foreground="#FF6347", font=("Helvetica", self.main_font_size, "bold"))
        self.main_frame.tag_configure("system", foreground="#808080", font=("Helvetica", self.system_font_size))
        self.main_frame.tag_configure("reward", foreground="#FF4500", font=("Helvetica", self.reward_font_size, "bold"))
        
        # Configure the same styles for the info frame
        self.info_frame.tag_configure("think", foreground="#6a0dad", font=("Helvetica", self.main_font_size, "italic"))
        self.info_frame.tag_configure("act", foreground="#008000", font=("Helvetica", self.main_font_size, "bold"))
        self.info_frame.tag_configure("speak", foreground="#0000CD", font=("Helvetica", self.main_font_size, "bold"))
        self.info_frame.tag_configure("observation", foreground="#B8860B", font=("Helvetica", self.main_font_size))
        self.info_frame.tag_configure("human", foreground="#FF6347", font=("Helvetica", self.main_font_size, "bold"))
        self.info_frame.tag_configure("system", foreground="#808080", font=("Helvetica", self.system_font_size, "bold"))
        self.info_frame.tag_configure("reward", foreground="#FF4500", font=("Helvetica", self.reward_font_size, "bold"))
        
    def setup_ui(self):
        # Create main panes with a nice modern look
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Main interaction display
        self.left_frame = ttk.Frame(self.paned_window, padding=5)
        self.paned_window.add(self.left_frame, weight=3)
        
        # Title for main interaction area
        self.main_title = ttk.Label(self.left_frame, text="Agent Interaction", font=("Helvetica", 14, "bold"))
        self.main_title.pack(pady=(0, 10), anchor="w")
        
        # Font size control for main frame
        self.font_frame = ttk.Frame(self.left_frame)
        self.font_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.font_label = ttk.Label(self.font_frame, text="Text Size:", font=("Helvetica", 10))
        self.font_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.decrease_font = ttk.Button(self.font_frame, text="-", width=3, command=self.decrease_font_size)
        self.decrease_font.pack(side=tk.LEFT, padx=(0, 2))
        
        self.increase_font = ttk.Button(self.font_frame, text="+", width=3, command=self.increase_font_size)
        self.increase_font.pack(side=tk.LEFT)
        
        # Main interaction display with scrollbar
        self.main_frame = scrolledtext.ScrolledText(self.left_frame, wrap=tk.WORD, height=30, 
                                                  font=("Helvetica", 11), bg="#ffffff")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Human input area
        self.input_frame = ttk.Frame(self.left_frame, padding=5)
        self.input_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.input_label = ttk.Label(self.input_frame, text="Your Response:", font=("Helvetica", 11, "bold"))
        self.input_label.pack(anchor="w", pady=(0, 5))
        
        self.input_text = ttk.Entry(self.input_frame, font=("Helvetica", 12))
        self.input_text.pack(fill=tk.X, side=tk.LEFT, expand=True)
        self.input_text.bind("<Return>", self.submit_human_input)
        
        self.submit_button = ttk.Button(self.input_frame, text="Submit", command=self.submit_human_input)
        self.submit_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Right panel - Game Info and Controls
        self.right_frame = ttk.Frame(self.paned_window, padding=5)
        self.paned_window.add(self.right_frame, weight=1)
        
        # Game info section
        self.info_title = ttk.Label(self.right_frame, text="Game Information", font=("Helvetica", 14, "bold"))
        self.info_title.pack(pady=(0, 10), anchor="w")
        
        # Font size control for info frame
        self.info_font_frame = ttk.Frame(self.right_frame)
        self.info_font_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_font_label = ttk.Label(self.info_font_frame, text="Info Size:", font=("Helvetica", 10))
        self.info_font_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.decrease_info_font = ttk.Button(self.info_font_frame, text="-", width=3, command=self.decrease_info_font_size)
        self.decrease_info_font.pack(side=tk.LEFT, padx=(0, 2))
        
        self.increase_info_font = ttk.Button(self.info_font_frame, text="+", width=3, command=self.increase_info_font_size)
        self.increase_info_font.pack(side=tk.LEFT)
        
        self.info_frame = scrolledtext.ScrolledText(self.right_frame, wrap=tk.WORD, height=15, 
                                                   font=("Helvetica", 11), bg="#ffffff")
        self.info_frame.pack(fill=tk.BOTH, expand=True)
        
        # Controls section
        self.controls_title = ttk.Label(self.right_frame, text="Controls", font=("Helvetica", 14, "bold"))
        self.controls_title.pack(pady=(10, 10), anchor="w")
        
        # Add the task selection dropdown
        self.task_frame = ttk.Frame(self.right_frame)
        self.task_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.task_label = ttk.Label(self.task_frame, text="Task Type:", font=("Helvetica", 11))
        self.task_label.pack(anchor="w", pady=(0, 5))
        
        self.task_dropdown = ttk.Combobox(self.task_frame, textvariable=self.current_task, state="readonly")
        self.task_dropdown.pack(fill=tk.X)
        
        # Config directory selection
        self.config_dir_frame = ttk.Frame(self.right_frame)
        self.config_dir_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.config_dir_label = ttk.Label(self.config_dir_frame, text="Config Directory:", font=("Helvetica", 11))
        self.config_dir_label.pack(anchor="w", pady=(0, 5))
        
        self.config_dir_entry = ttk.Entry(self.config_dir_frame, font=("Helvetica", 10))
        self.config_dir_entry.insert(0, self.configs_root_dir)
        self.config_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.config_dir_button = ttk.Button(self.config_dir_frame, text="Scan", command=self.scan_config_dir)
        self.config_dir_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        self.controls_frame = ttk.Frame(self.right_frame)
        self.controls_frame.pack(fill=tk.X)
        
        self.start_button = ttk.Button(self.controls_frame, text="Start New Game", 
                                      command=self.start_new_game)
        self.start_button.pack(fill=tk.X, pady=5)
        
        self.clear_button = ttk.Button(self.controls_frame, text="Clear Display", 
                                      command=self.clear_display)
        self.clear_button.pack(fill=tk.X, pady=5)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W, font=("Helvetica", 10))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Disable input initially
        self.toggle_input(False)
        
    def scan_config_dir(self):
        """Update the config directory and scan for task folders"""
        new_dir = self.config_dir_entry.get().strip()
        if new_dir:
            self.configs_root_dir = new_dir
            self.load_task_folders()
            self.message_queue.put(('system', f"Scanned config directory: {new_dir}"))
            
    def increase_font_size(self):
        self.main_font_size += 1
        self.reward_font_size += 1
        self.system_font_size += 1
        self.apply_font_sizes(self.main_frame)
        
    def decrease_font_size(self):
        if self.main_font_size > 8:  # Don't let it get too small
            self.main_font_size -= 1
            self.reward_font_size -= 1
            self.system_font_size -= 1
            self.apply_font_sizes(self.main_frame)
            
    def increase_info_font_size(self):
        self.main_font_size += 1
        self.reward_font_size += 1
        self.system_font_size += 1
        self.apply_font_sizes(self.info_frame)
        
    def decrease_info_font_size(self):
        if self.main_font_size > 8:  # Don't let it get too small
            self.main_font_size -= 1
            self.reward_font_size -= 1
            self.system_font_size -= 1
            self.apply_font_sizes(self.info_frame)
            
    def apply_font_sizes(self, frame):
        # Update all the font tags with new sizes
        frame.tag_configure("think", font=("Helvetica", self.main_font_size, "italic"))
        frame.tag_configure("act", font=("Helvetica", self.main_font_size, "bold"))
        frame.tag_configure("speak", font=("Helvetica", self.main_font_size, "bold"))
        frame.tag_configure("observation", font=("Helvetica", self.main_font_size))
        frame.tag_configure("human", font=("Helvetica", self.main_font_size, "bold"))
        frame.tag_configure("system", font=("Helvetica", self.system_font_size))
        frame.tag_configure("reward", font=("Helvetica", self.reward_font_size, "bold"))
        
        # Also update the main text widget's default font
        current_font = frame['font']
        if isinstance(current_font, str):
            # Parse font string if it's a string
            font_parts = current_font.split()
            family = font_parts[0] if len(font_parts) > 0 else "Helvetica"
            frame.configure(font=(family, self.main_font_size))
        
    def toggle_input(self, enabled=True):
        state = "normal" if enabled else "disabled"
        self.input_text.configure(state=state)
        self.submit_button.configure(state=state)
        
    def submit_human_input(self, event=None):
        if self.waiting_for_human_input:
            response = self.input_text.get()
            if response:
                self.main_frame.insert(tk.END, f"👤 [Human] {response}\n", "human")
                self.main_frame.see(tk.END)
                self.input_text.delete(0, tk.END)
                self.human_input_queue.put(response)
                self.waiting_for_human_input = False
                self.toggle_input(False)
                
    def process_queue(self):
        try:
            while True:
                msg_type, message = self.message_queue.get_nowait()
                
                if msg_type == 'log':
                    # We're no longer using this type for the info frame
                    pass
                
                elif msg_type == 'think':
                    # Format think messages with a lightbulb emoji
                    self.main_frame.insert(tk.END, f"💭 [Think] {message}\n", "think")
                    self.main_frame.see(tk.END)
                
                elif msg_type == 'act':
                    # Format action messages with a gear emoji
                    self.main_frame.insert(tk.END, f"⚙️ [Action] {message}\n", "act")
                    self.main_frame.see(tk.END)
                
                elif msg_type == 'speak':
                    # Format speak messages with a speech bubble emoji
                    self.main_frame.insert(tk.END, f"💬 [Agent] {message}\n", "speak")
                    self.main_frame.see(tk.END)
                    self.waiting_for_human_input = True
                    self.toggle_input(True)
                    self.input_text.focus()
                
                elif msg_type == 'observation':
                    # Format observation messages with an eye emoji
                    self.main_frame.insert(tk.END, f"👁️ [Observation] {message}\n", "observation")
                    self.main_frame.see(tk.END)
                
                elif msg_type == 'system':
                    # Format system messages with an info emoji but make them less prominent
                    self.main_frame.insert(tk.END, f"ℹ️ {message}\n", "system")
                    self.main_frame.see(tk.END)
                    
                elif msg_type == 'reward':
                    # Format reward messages with a trophy emoji
                    self.main_frame.insert(tk.END, f"🏆 {message}\n", "reward")
                    self.main_frame.see(tk.END)
                    
                elif msg_type == 'status':
                    self.status_bar.config(text=message)
                
                self.message_queue.task_done()
                
        except queue.Empty:
            pass
        
        self.root.after(100, self.process_queue)
        
    def clear_display(self):
        self.main_frame.delete(1.0, tk.END)
        self.info_frame.delete(1.0, tk.END)
        
    def start_new_game(self):
        self.clear_display()
        
        selected_task = self.current_task.get()
        if selected_task == "No tasks found":
            self.message_queue.put(('system', "Error: No valid task configurations found. Please check the config directory."))
            return
            
        self.message_queue.put(('system', f"Starting new game with task type: {selected_task}"))
        self.message_queue.put(('status', "Game in progress"))
        
        # Start the game in a separate thread
        threading.Thread(target=self.run_game, daemon=True).start()
        
    def run_game(self):
        # Load configs
        try:
            selected_task = self.current_task.get()
            
            # Construct paths to the config files based on selected task
            config_path = os.path.join(self.configs_root_dir, selected_task)
            configs_file = os.path.join(config_path, 'base_config.yaml')
            configs_file2 = os.path.join(config_path, 'base_config2.yaml')
            
            # Check if the config files exist
            if not os.path.exists(configs_file) or not os.path.exists(configs_file2):
                self.message_queue.put(('system', f"Error: Config files not found in {config_path}"))
                self.message_queue.put(('status', "Error: Config files not found"))
                return
                
            # Load the config files
            config = load_yaml(configs_file)
            config2 = load_yaml(configs_file2)
            
            # Setup environments
            split = "eval_out_of_distribution"
            
            # Clear the info frame first
            self.info_frame.delete(1.0, tk.END)
            
            # Display task info in the info frame
            self.info_frame.insert(tk.END, f"SELECTED TASK: {selected_task.upper()}\n\n", "system")
            self.info_frame.insert(tk.END, f"Config Path: {config_path}\n\n", "system")
            
            # Setup environments
            env1 = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
            env1 = env1.init_env(batch_size=1)
            
            env2 = getattr(alfworld.agents.environment, config2["env"]["type"])(config2, train_eval=split)
            env2 = env2.init_env(batch_size=1)
            
            # Load prompts
            main_prompt = "prompts/main_prompt_respact.txt"
            respact_prompt_file_path = "prompts/respact_alfred_old.json"
            main = load_txt(main_prompt)
            respact_prompts = load_json(respact_prompt_file_path)
            
            # Prefixes dictionary
            prefixes = {
                'pick_and_place': 'put',
                'pick_clean_then_place': 'clean',
                'pick_heat_then_place': 'heat',
                'pick_cool_then_place': 'cool',
                'look_at_obj': 'examine',
                'pick_two_obj': 'puttwo'
            }
            
            # Reset environments
            ob1, info1 = env1.reset()
            ob2, info2 = env2.reset()
            
            ob1 = '\n'.join(ob1[0].split('\n\n')[1:])
            name1 = '/'.join(info1['extra.gamefile'][0].split('/')[-3:-1])
            ob2 = '\n'.join(ob2[0].split('\n\n')[1:])
            name2 = '/'.join(info2['extra.gamefile'][0].split('/')[-3:-1])
            
            ob = ob2
            info = info2
            name = name2
            
            game_file = info['extra.gamefile'][0]
            oracle_info = oracle_support(game_file)
            
            # Extract task description and objects in the observation
            task_desc = "Unknown task"
            objects_list = []
            
            if 'Your task is to:' in ob:
                task_desc = ob.split('Your task is to: ')[1].strip()
                
                # Generate human task description for info panel as well
                human_task_desc = generate_human_task_description(task_desc)
                
                # Add the formatted task description to the main interaction panel queue
                self.message_queue.put(('human', human_task_desc))
            else:
                human_task_desc = "Unknown task"
            
            # Extract objects from observation
            if 'you see' in ob.lower():
                objects_text = ob.lower().split('you see ')[1].split('.')[0]
                objects = objects_text.split(', ')
                if objects and len(objects) > 0:
                    # Last item might have "and a" instead of just "a"
                    if objects[-1].startswith('and '):
                        objects[-1] = objects[-1][4:]
                    
                    objects_list = [obj.strip() for obj in objects if obj.strip()]
            
            # Display information in the game info panel
            self.info_frame.insert(tk.END, "Original Task Description:\n", "system")
            self.info_frame.insert(tk.END, f"{task_desc}\n\n", "act")
            
            self.info_frame.insert(tk.END, "Human Instructions:\n", "system")
            self.info_frame.insert(tk.END, f"{human_task_desc}\n\n", "human")
            
            self.info_frame.insert(tk.END, "Oracle Information:\n", "system")
            self.info_frame.insert(tk.END, f"{oracle_info}\n\n", "observation")
            
            self.info_frame.insert(tk.END, "Objects in the Environment:\n", "system")
            for obj in objects_list:
                self.info_frame.insert(tk.END, f"• {obj}\n", "think")
            
            # Determine game type and run
            # First try to match the selected task directly with prefixes
            prompt_key = None
            if selected_task in prefixes:
                prompt_key = prefixes[selected_task]
            else:
                # If not a direct match, try to find a prefix that matches the beginning of the task name
                for k, v in prefixes.items():
                    if name.startswith(k):
                        prompt_key = v
                        break
            
            if prompt_key:
                prompt = main + 'Interact with a household to solve a task. Here are 2 examples.\n' + respact_prompts[f'respact_{prompt_key}_0'] + respact_prompts[f'respact_{prompt_key}_1'] + '\nHere is the task.\n'
                
                r = self.alfworld_run(prompt, oracle_info, env1, env2, ob=ob)
                
                self.message_queue.put(('reward', f"Final reward: {r}\n"))
                self.message_queue.put(('status', f"Game completed with reward: {r}"))
            else:
                self.message_queue.put(('system', f"Error: Could not determine prompt type for task: {selected_task}"))
                self.message_queue.put(('status', "Error: Unknown task type"))
                    
        except Exception as e:
            self.message_queue.put(('system', f"Error: {str(e)}\n"))
            self.message_queue.put(('status', "Error occurred"))
            
    def alfworld_run(self, prompt, oracle_info, env1, env2, to_print=True, ob=''):
        user_simulator_prompt = 'prompts/user_sim.txt'
        user_simulator = LLMUserAgent(oracle_info, 'gpt4', user_simulator_prompt)
        
        init_prompt = prompt + ob + '\n>'
        running_prompt = ''
        
        # Display initial observation
        self.message_queue.put(('system', "Game started. Agent is exploring the environment..."))
        self.message_queue.put(('observation', ob))
        
        # Extract and generate human task description
        task_desc = "Unknown task"
        if 'Your task is to:' in ob:
            task_desc = ob.split('Your task is to: ')[1].strip()
            
            # Generate a more natural human description
            human_task_desc = generate_human_task_description(task_desc)
            
            # Add the human task description to the interaction
            self.message_queue.put(('human', human_task_desc))
            
            # Update prompt to include human instruction
            running_prompt += f' Human: {human_task_desc}\n>'
        
        for i in range(1, 50):
            # Update status bar with current turn
            self.status_bar.config(text=f"Turn {i}: Agent thinking...")
            self.root.update_idletasks()
            
            message = [
                {"role":"system", "content":init_prompt + running_prompt}
            ]
            action = llm(message, stop=['\n']).strip()
            _action = action.lstrip('> ')
            done = False
            reward = 0
            
            if _action.startswith('think:'):
                self.status_bar.config(text=f"Turn {i}: Agent thinking")
                self.message_queue.put(('think', _action[6:].strip()))
                observation = 'OK.'
            elif _action.startswith('speak:'):
                self.status_bar.config(text=f"Turn {i}: Agent asking a question")
                question = _action[6:].strip()
                self.message_queue.put(('speak', question))
                
                # Wait for human input
                self.waiting_for_human_input = True
                self.input_text.focus_set()
                
                while self.human_input_queue.empty() and self.waiting_for_human_input:
                    self.status_bar.config(text=f"Turn {i}: Waiting for your response...")
                    time.sleep(0.1)
                    self.root.update()  # Keep UI responsive while waiting
                    
                if not self.human_input_queue.empty():
                    user_response = self.human_input_queue.get()
                    observation = f"Human: {user_response}"
                else:
                    observation = "Human: (No response)"
                    
            elif _action.startswith('act:'):
                self.status_bar.config(text=f"Turn {i}: Agent performing action")
                act_cmd = _action[4:].strip()
                self.message_queue.put(('act', act_cmd))
                
                # Process the action in both environments
                try:
                    _,_,_,_ = env1.step([act_cmd])
                    observation, reward, done, info = env2.step([act_cmd])
                    observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
                except Exception as e:
                    self.message_queue.put(('system', f"Error executing action: {str(e)}"))
                    observation = f"Error: Could not execute action. {str(e)}"
            else:
                self.status_bar.config(text=f"Turn {i}: Agent sent invalid format")
                self.message_queue.put(('system', f"Invalid action format: {_action}"))
                observation = 'Wrong response format. Please specify if it is a think, speak or act response.'
                
            self.message_queue.put(('observation', observation))
            
            running_prompt += f' {action}\n{observation}\n>'
            
            if done:
                self.status_bar.config(text=f"Game completed with reward: {reward}")
                self.message_queue.put(('system', "Task completed!"))
                return reward
                
        self.status_bar.config(text="Game ended - max turns reached")
        return 0

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = AlfworldGUI(root)
    root.mainloop()