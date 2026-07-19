import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

load_dotenv()

@CrewBase
class AiLeague():
    agents_config = "config/agents.yaml"
    tasks_config  = "config/tasks.yaml"

    def __init__(self) -> None:
        # LLM Configurations for CrewAI
        self.thinker_llm = LLM(
            model=f"ollama/{os.getenv('OLLAMA_MODEL', 'qwen3:14b')}",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            temperature=0.85,
        )
        self.evaluator_llm = LLM(
            model=f"ollama/{os.getenv('OLLAMA_EVAL_MODEL', 'qwen3:8b')}",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            temperature=0.6,
        )

    def _load_tools(self):
        """Registers all tools into the Agent toolbelt."""
        tools = []
        try:
            from tools.web_search      import web_search_tool
            from tools.file_reader     import file_reader_tool
            from tools.file_writer     import file_writer_tool
            from tools.code_executor   import code_executor_tool
            from tools.system_context  import system_context_tool
            from tools.notepad_tool    import notepad_read_tool, notepad_write_tool
            
            tools.extend([
                web_search_tool, 
                file_reader_tool, 
                file_writer_tool,
                code_executor_tool,
                system_context_tool,
                notepad_read_tool,
                notepad_write_tool
            ])
        except Exception as e:
            print(f"[Crew] Tool load failed: {e}")
        return tools

    @agent
    def silicon_spirit(self) -> Agent:
        """Agent A — Primary thinker (qwen3:14b)"""
        return Agent(
            config=self.agents_config["silicon_spirit"],
            llm=self.thinker_llm,
            verbose=True,
            allow_delegation=False,
            memory=True,
            tools=self._load_tools(),
        )

    @agent
    def evil_neuro_evaluator(self) -> Agent:
        """Agent B — Evaluator/judge (qwen3:8b)"""
        return Agent(
            config=self.agents_config["evil_neuro_evaluator"],
            llm=self.evaluator_llm,
            verbose=True,
            allow_delegation=False,
            memory=False,
            tools=[],
        )

    @task
    def general_chat_task(self) -> Task:
        return Task(
            config=self.tasks_config["general_chat_task"],
            agent=self.silicon_spirit(),
        )

    @task
    def evaluate_response_task(self) -> Task:
        return Task(
            config=self.tasks_config["evaluate_response_task"],
            agent=self.evil_neuro_evaluator(),
            context=[self.general_chat_task()],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Silicon Spirit league crew"""
        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )