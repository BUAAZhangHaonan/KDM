"""Prompts for the three abstention styles (task v2 section 4)."""

STYLE1 = ("Look at the image. Name the main object shown in the image. "
          "Reply with a single English noun (one word or a short compound). "
          "If you do not recognize the object or are not sure, reply exactly: UNKNOWN")

STYLE2 = ("Look at the image. First state whether you can recognize the main object, "
          "by writing RECOGNIZED or NOT_RECOGNIZED, then after a comma give the object's "
          "name as a single English noun. Example: RECOGNIZED, pizza. "
          "If you cannot recognize it, reply exactly: NOT_RECOGNIZED, UNKNOWN")

STYLE3_ROUND_A = ("Look at the image. Can you recognize the main object in it? "
                  "Answer with exactly one word: YES or NO.")

# round B of style 3 is STYLE1 (naming with UNKNOWN marker)

STYLES = {'style1': [STYLE1], 'style2': [STYLE2], 'style3': [STYLE3_ROUND_A, STYLE1]}


def existence_prompt(cat_name):
    import re
    nice = re.sub(r'\([^)]*\)', '', cat_name).replace('_', ' ').strip()
    return (f"Look at the image. Is there a {nice} in the image? "
            "Answer with exactly one word: Yes or No.")
