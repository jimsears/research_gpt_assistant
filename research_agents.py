
"""
AI Research Agents for Specialized Tasks

Implements:
1. Summarizer Agent - Document summarization
2. QA Agent - Question answering
3. Research Workflow Agent - End to end research sessions
4. Agent Orchestrator - Simple routing across agents
"""

import json
import time
from typing import List, Dict, Any, Tuple


class BaseAgent:
    def __init__(self, research_assistant):
        """
        Base class for all research agents.

        Stores a reference to the research assistant which must expose:
          - doc_processor with:
              - documents: Dict[doc_id, {"chunks": List[str], ...}]
              - find_similar_chunks(query, top_k) -> List[Tuple[str, float, str]]
          - _call_mistral(prompt: str) -> str
          - chain_of_thought_reasoning(question, relevant_chunks) -> str
        """
        self.assistant = research_assistant
        self.agent_name = "BaseAgent"

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Interface method all agents must implement."""
        raise NotImplementedError("Each agent must implement execute_task")


class SummarizerAgent(BaseAgent):
    def __init__(self, research_assistant):
        super().__init__(research_assistant)
        self.agent_name = "SummarizerAgent"
        # limits to keep prompts within token budgets
        self.max_chunks = 40
        self.sleep_between_calls = 0.3

    def _get_document_text(self, doc_id: str) -> str:
        """
        Safely retrieve concatenated document text from the document processor.
        Returns a concatenation of the first N chunks to respect token limits.
        """
        docs = getattr(self.assistant.doc_processor, "documents", {})
        if doc_id not in docs:
            return ""
        chunks = docs[doc_id].get("chunks", [])
        if not chunks:
            return ""
        return "\n\n".join(chunks[: self.max_chunks])

    def summarize_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Summarize a specific document.

        Steps:
          1. Retrieve document chunks
          2. Create summarization prompt
          3. Generate summary using the LLM
          4. Return structured summary
        """
        document_text = self._get_document_text(doc_id)
        if not document_text:
            return {
                "doc_id": doc_id,
                "summary": "No text available for this document.",
                "word_count": 0,
                "key_topics": [],
            }

        # Clear instruction to summarize the content, not return a template
        summary_prompt = f"""
You are an expert scientific editor. Summarize the following academic paper clearly and concisely.

Requirements:
- Focus only on the content of the paper text below. Do not return a prompt template.
- Include: (1) main research question or hypothesis, (2) methodology, (3) key findings,
  (4) conclusions, and (5) limitations.
- Use plain language. Target 150 to 250 words.
- If the text appears unrelated or insufficient, say so briefly and end.

Paper ID: {doc_id}

Paper Text (truncated):
\"\"\"
{document_text}
\"\"\"
""".strip()

        time.sleep(self.sleep_between_calls)
        summary = self.assistant._call_mistral(summary_prompt) or ""

        # Placeholder for future topic extraction
        key_topics: List[str] = []

        return {
            "doc_id": doc_id,
            "summary": summary.strip(),
            "word_count": len(summary.split()),
            "key_topics": key_topics,
        }

    def create_literature_overview(self, doc_ids: List[str]) -> Dict[str, Any]:
        """
        Create a short literature overview from multiple documents.
        """
        individual_summaries: List[Dict[str, Any]] = []
        for doc_id in doc_ids:
            try:
                time.sleep(self.sleep_between_calls)
                s = self.summarize_document(doc_id)
                individual_summaries.append(s)
            except Exception as e:
                individual_summaries.append(
                    {
                        "doc_id": doc_id,
                        "summary": f"Error summarizing document: {e}",
                        "word_count": 0,
                        "key_topics": [],
                    }
                )

        overview_prompt = f"""
You are an expert scientific editor. Given the individual paper summaries below,
write a literature overview that identifies:
- Common research themes
- Different methodological approaches
- Consistent findings and contradictions
- Research gaps
- Short list of future research directions

Keep it concise (250 to 400 words). Do not output a prompt template.

Individual Summaries (JSON):
{json.dumps(individual_summaries, indent=2)}
""".strip()

        time.sleep(self.sleep_between_calls)
        overview = self.assistant._call_mistral(overview_prompt) or ""

        return {
            "overview": overview.strip(),
            "papers_analyzed": len(doc_ids),
            "individual_summaries": individual_summaries,
        }

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        if "doc_id" in task_input:
            return self.summarize_document(task_input["doc_id"])
        if "doc_ids" in task_input:
            return self.create_literature_overview(task_input["doc_ids"])
        return {"error": "Invalid input for SummarizerAgent. Provide doc_id or doc_ids."}


class QAAgent(BaseAgent):
    def __init__(self, research_assistant):
        super().__init__(research_assistant)
        self.agent_name = "QAAgent"
        self.top_k = 5
        self.sleep_between_calls = 0.3

    def _format_chunks_for_prompt(self, chunks: List[Tuple[str, float, str]]) -> str:
        """
        Format retrieved chunks for inclusion in the prompt.
        Expects chunks as list of (chunk_text, similarity_score, doc_id).
        """
        lines: List[str] = []
        for i, (txt, score, did) in enumerate(chunks, 1):
            lines.append(f"[{i}] (doc_id={did}, score={score:.3f})\n{txt}\n")
        return "\n".join(lines)

    def answer_factual_question(self, question: str) -> Dict[str, Any]:
        """
        Answer factual questions using only evidence from retrieved chunks.
        """
        relevant_chunks = self.assistant.doc_processor.find_similar_chunks(
            question, top_k=min(3, self.top_k)
        )

        qa_context = self._format_chunks_for_prompt(relevant_chunks)
        qa_prompt = f"""
You are a precise research assistant. Answer the factual question using ONLY the evidence below.
If the evidence is insufficient or does not contain the answer, reply exactly with "Insufficient evidence."

Question:
{question}

Evidence:
{qa_context}

Instructions:
- Provide a concise answer of one to three sentences.
- Cite sources by their [index] numbers, for example [1], [2].
- Do not fabricate details beyond the evidence.
""".strip()

        time.sleep(self.sleep_between_calls)
        answer = self.assistant._call_mistral(qa_prompt) or ""

        # Build a simple source list that includes doc ids
        sources = []
        for idx, (_txt, _score, did) in enumerate(relevant_chunks, 1):
            sources.append({"index": idx, "doc_id": did})

        return {
            "question": question,
            "answer": answer.strip(),
            "sources": sources,
            "confidence": "high"
            if answer.strip() and "Insufficient evidence" not in answer
            else "low",
        }

    def answer_analytical_question(self, question: str) -> Dict[str, Any]:
        """
        Answer analytical questions with chain of thought reasoning.
        """
        relevant = self.assistant.doc_processor.find_similar_chunks(
            question, top_k=self.top_k
        )
        time.sleep(self.sleep_between_calls)
        response = self.assistant.chain_of_thought_reasoning(question, relevant) or ""

        srcs = []
        for idx, (_txt, _score, did) in enumerate(relevant, 1):
            srcs.append({"index": idx, "doc_id": did})

        return {
            "question": question,
            "analysis": response.strip(),
            "reasoning_type": "chain_of_thought",
            "sources": srcs,
        }

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        q = task_input.get("question", "").strip()
        qtype = task_input.get("type", "factual")
        if not q:
            return {"error": "QAAgent requires a 'question' field."}
        if qtype == "analytical":
            return self.answer_analytical_question(q)
        return self.answer_factual_question(q)


class ResearchWorkflowAgent(BaseAgent):
    def __init__(self, research_assistant):
        super().__init__(research_assistant)
        self.agent_name = "ResearchWorkflowAgent"
        self.summarizer = SummarizerAgent(research_assistant)
        self.qa_agent = QAAgent(research_assistant)
        self.sleep_between_steps = 0.3

    def conduct_research_session(self, research_topic: str) -> Dict[str, Any]:
        """
        Conduct a simple end to end research session on a topic.
        """
        session_results: Dict[str, Any] = {
            "research_topic": research_topic,
            "generated_questions": [],
            "document_analysis": {},
            "answers": [],
            "research_gaps": "",
            "future_directions": "",
        }

        # Step 1: generate questions
        questions_prompt = f"""
You are a senior researcher. Generate four specific, answerable research questions about the topic below.
Cover different aspects: definition or what, methods or how, cause or why, and limits or open problems.
Return only the questions as a simple numbered list.

Topic: {research_topic}
""".strip()
        time.sleep(self.sleep_between_steps)
        generated_questions_text = self.assistant._call_mistral(questions_prompt) or ""
        questions = [q.strip("- ").strip() for q in generated_questions_text.split("\n") if q.strip()]
        session_results["generated_questions"] = questions[:4]

        # Step 2: find similar chunks and collect doc ids
        relevant_chunks = self.assistant.doc_processor.find_similar_chunks(
            research_topic, top_k=10
        )
        doc_ids_ordered = list(dict.fromkeys([doc_id for (_t, _s, doc_id) in relevant_chunks]))

        # Step 3: multi document overview
        if doc_ids_ordered:
            time.sleep(self.sleep_between_steps)
            overview = self.summarizer.create_literature_overview(doc_ids_ordered)
            session_results["document_analysis"] = overview

        # Step 4: answer questions
        answers: List[Dict[str, Any]] = []
        for q in session_results["generated_questions"]:
            if not q:
                continue
            factual = self.qa_agent.answer_factual_question(q)
            if "Insufficient evidence" in factual.get("answer", ""):
                analytical = self.qa_agent.answer_analytical_question(q)
                answers.append({"question": q, "mode": "analytical", **analytical})
            else:
                answers.append({"question": q, "mode": "factual", **factual})
            time.sleep(self.sleep_between_steps)
        session_results["answers"] = answers

        # Step 5: gaps and directions
        gaps_prompt = f"""
You are a senior researcher. Based on the topic and the literature overview below,
identify three to five research gaps and propose three to five concrete future research directions.

Topic:
{research_topic}

Literature Overview (JSON):
{json.dumps(session_results.get("document_analysis", {}), indent=2)}
""".strip()
        time.sleep(self.sleep_between_steps)
        gaps_and_directions = self.assistant._call_mistral(gaps_prompt) or ""
        session_results["research_gaps"] = gaps_and_directions.strip()

        return session_results

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        if "research_topic" in task_input:
            return self.conduct_research_session(task_input["research_topic"])
        return {"error": "Invalid input for ResearchWorkflowAgent. Provide research_topic."}


class AgentOrchestrator:
    def __init__(self, research_assistant):
        """
        Orchestrates multiple agents for complex tasks.
        """
        self.assistant = research_assistant
        self.agents = {
            "summarizer": SummarizerAgent(research_assistant),
            "qa": QAAgent(research_assistant),
            "workflow": ResearchWorkflowAgent(research_assistant),
        }

    def route_task(self, task_type: str, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route tasks to an agent by type: 'summarizer', 'qa', or 'workflow'.
        """
        agent = self.agents.get(task_type)
        if not agent:
            return {"error": f"Unknown task type: {task_type}"
                   }
        return agent.execute_task(task_input)

    def execute_complex_workflow(self, workflow_description: str) -> Dict[str, Any]:
        """
        Execute a simple parsed workflow from a natural language description.

        Very lightweight parser that looks for keywords:
          - "summarize \"DOC_ID\""
          - "answer QUESTION"
          - "research TOPIC"
        Steps are executed in the order they are detected.
        """
        results: Dict[str, Any] = {
            "workflow_description": workflow_description,
            "steps_executed": [],
            "final_result": {},
        }

        text = workflow_description.strip().lower()

        # detect a summarize step
        if "summarize" in text:
            import re
            m = re.search(r'summarize\s+"([^"]+)"', workflow_description, flags=re.IGNORECASE)
            doc_id = m.group(1).strip() if m else None
            if doc_id:
                s = self.agents["summarizer"].execute_task({"doc_id": doc_id})
                results["steps_executed"].append({"step": "summarize", "doc_id": doc_id})
                results.setdefault("intermediate", {})["summary"] = s

        # detect a research session step
        if "research" in text:
            # naive approach: everything after the word "research"
            topic = workflow_description.split("research", 1)[-1].strip(" :")
            if topic:
                wf = self.agents["workflow"].execute_task({"research_topic": topic})
                results["steps_executed"].append({"step": "research", "topic": topic})
                results.setdefault("intermediate", {})["research_session"] = wf

        # detect a question to answer
        if "answer" in text or "question" in text:
            segment = None
            if "answer" in text:
                segment = workflow_description.split("answer", 1)[-1].strip(" :?")
            elif "question" in text:
                segment = workflow_description.split("question", 1)[-1].strip(" :?")
            if segment:
                qa = self.agents["qa"].execute_task({"question": segment, "type": "factual"})
                results["steps_executed"].append({"step": "qa", "question": segment})
                results.setdefault("intermediate", {})["qa"] = qa

        # compact roll up
        if "research_session" in results.get("intermediate", {}):
            results["final_result"] = results["intermediate"]["research_session"]
        elif "summary" in results.get("intermediate", {}):
            results["final_result"] = results["intermediate"]["summary"]
        elif "qa" in results.get("intermediate", {}):
            results["final_result"] = results["intermediate"]["qa"]
        else:
            results["final_result"] = {"note": "No actionable steps detected."}

        return results
