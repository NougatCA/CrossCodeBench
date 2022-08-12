
category_to_task_type = {
    "Detection -> Defect Detection": "Classification -> Binary",
    "Detection -> Clone Detection": "Classification -> Binary",
    "Classification -> Verification -> Docstring Verification": "Classification -> Binary",
    "Classification -> Verification -> Code Verification": "Classification -> Binary",
    "Detection -> Defect Detection -> Buffer Error Vulnerability": "Classification -> Binary",
    "Detection -> Defect Detection -> Resource Management Vulnerability": "Classification -> Binary",
    "Detection -> Defect Detection -> API Call Vulnerability": "Classification -> Binary",
    "Detection -> Defect Detection -> Arithmetic Vulnerability": "Classification -> Binary",
    "Detection -> Defect Detection -> Array Usage Vulnerability": "Classification -> Binary",
    "Detection -> Defect Detection -> Pointer Vulnerability": "Classification -> Binary",
    "Detection -> Defect Detection -> Buffer Overrun": "Classification -> Binary",
    "Classification -> Comment Type": "Classification -> Multi-label",
    "Detection -> Bug Type": "Classification -> Multi-label",
    "Fill in the blank -> Exception Type": "Classification -> Multi-label",

    "Fill in the blank -> Code expression": "Fill in the blank",
    "Fill in the blank -> Method Name": "Fill in the blank",
    "Fill in the blank -> Assert Statement": "Fill in the blank",

    "Translation": "Translation",

    "Code Modification -> Bug Fixing": "Generation -> Rewrite",
    "Code Modification -> Mutation Injection": "Generation -> Rewrite",
    "Code Generation": "Generation -> Text-to-Code",
    "Generation -> Commit Message": "Generation -> Code-to-Text",
    "Generation -> Pseudo-Code": "Generation -> Text-to-Code",
    "Generation": "Generation -> Text-to-Code",
    "Generation -> Program Synthesis": "Generation -> Text-to-Code",
    "Generation -> API Sequence": "Generation -> Text-to-Code",

    "Summarization": "Summarization",

    "Named Entity Recognition -> Type Prediction": "Type Prediction",

    "Question Answering": "Question Answering"
}


def convert_category_to_task_type(category):
    return category_to_task_type[category]
