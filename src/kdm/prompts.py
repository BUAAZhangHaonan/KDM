"""Frozen task prompts. Guided abstention is the main protocol."""
MARKERS=("UNKNOWN","UNCLEAR","UNSURE","I cannot identify it")


def task_prompt(question,marker="UNKNOWN",guided=True,attempt=False):
    if not isinstance(question,str) or not question.strip(): raise ValueError("Missing question")
    if attempt:
        return question.strip()+"\nGive your best estimate as a specific short answer."
    prompt=question.strip()+"\nGive a concise answer."
    if guided:
        if marker not in MARKERS: raise ValueError("Unregistered abstention expression")
        prompt+=f'\nIf you cannot determine the answer reliably, respond with "{marker}".'
    return prompt


def closed_prompt(question,names):
    if len(names)!=len(set(names)) or not names: raise ValueError("Invalid candidate list")
    return (question.strip()+"\nChoose the best matching name from the following list. "
            "Reply with the name only.\n"+"; ".join(names))


def reference_prompt(main,reference_marker=None,remove_guide=False):
    """A caller supplies the question, rather than editing an already-built chat string."""
    return task_prompt(main,reference_marker or "UNKNOWN",guided=not remove_guide)
