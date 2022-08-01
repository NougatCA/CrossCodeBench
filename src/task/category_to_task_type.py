
category_to_task_type = {
    "Detection -> Defect Detection": "Classification",
    "Detection -> Clone Detection": "Classification",
    "Fill in the blank -> Exception Type": "Fill in the blank",
    "Classification -> Verification -> Docstring Verification": "Classification",
    "Classification -> Verification -> Code Verification": "Classification",
    "Translation": "Translation",
    "Code Modification -> Bug Fixing": "Generation",
    "Fill in the blank -> Code expression": "Fill in the blank",
    "Code Modification -> Mutation Injection": "Generation",
    "Fill in the blank -> Assert Statement": "Generation",
    "Summarization": "Summarization",
    "Code Generation": "Generation",
    "Named Entity Recognition -> Type Prediction": "Type Prediction",
    "Question Answering": "Question Answering",
    "Detection -> Defect Detection -> Buffer Error Vulnerability": "Classification",
    "Detection -> Defect Detection -> Resource Management Vulnerability": "Classification",
    "Detection -> Defect Detection -> API Call Vulnerability": "Classification",
    "Detection -> Defect Detection -> Arithmetic Vulnerability": "Classification",
    "Detection -> Defect Detection -> Array Usage Vulnerability": "Classification",
    "Detection -> Defect Detection -> Pointer Vulnerability": "Classification",
    "Detection -> Defect Detection -> Buffer Overrun": "Classification",
    "Generation -> Commit Message": "Generation",
    "Classification -> Comment Type": "Classification",
    "Generation -> Pseudo-Code": "Generation",
    "Fill in the blank -> Method Name": "Fill in the blank",
    "Generation": "Generation",
    "Generation -> Program Synthesis": "Generation",
    "Generation -> API Sequence": "Generation",
    "Detection -> Bug Type": "Classification"
}


def convert_category_to_task_type(category):
    return category_to_task_type[category]
