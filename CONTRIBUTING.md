

# Contributing to Phishing Website Detection

First off, thank you for considering contributing to this project! It’s people like you who help make the internet a safer place. 

As a project focused on **Cybersecurity** and **Machine Learning**, we maintain high standards for code quality, model transparency, and ethical research.

---

##  Table of Contents
- [Code of Conduct](#-code-of-conduct)
- [How Can I Contribute?](#-how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Code Contributions](#code-contributions)
- [ Technical Standards](#-technical-standards)
- [ Ethical AI Guidelines](#️-ethical-ai-guidelines)
- [ Pull Request Process](#-pull-request-process)

---

##  Code of Conduct
By participating in this project, you agree to abide by our standards of professional and respectful communication. We focus on constructive feedback and empathy within the community.

---

##  How Can I Contribute?

### Reporting Bugs
If you find a bug in the feature extraction logic or a crash in the GUI:
1. Search the **Issues** tab to see if it’s already been reported.
2. If not, open a new issue using a clear title.
3. Provide a **Reproducible Example** (e.g., the specific URL that caused the error) and your environment details.

### Suggesting Enhancements
We are always looking to improve our detection accuracy. Suggestions for new features are welcome, particularly:
- **New URL Features:** Additional lexical or structural indicators of phishing.
- **Model Optimization:** Hyperparameter tuning or new ensemble methods.
- **Explainable AI (XAI):** Better ways to visualize why the AI flagged a URL.

---

##  Technical Standards

To maintain a production-grade codebase (MLOps), please follow these guidelines:

1. **Pythonic Code:** Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines.
2. **Type Hinting:** Use Python type hints (e.g., `def analyze(url: str) -> float:`) to ensure code clarity.
3. **Docstrings:** Every function must have a Google-style docstring explaining its purpose, parameters, and return values.
4. **Reproducibility:** If you update the model, ensure you also update the `scaler.pkl` and `training_features_list.pkl` so others can run your code.

---

##  Ethical AI Guidelines

This is a **Security-AI** project. All contributions must respect our ethical framework:

- **Non-Maleficence:** Contributions must not be designed to help attackers bypass filters. This tool is for **defensive** research.
- **Explainability (XAI):** We prioritize models that are transparent. If you suggest a "Black Box" model (like a Deep Neural Network), you must also suggest an interpretation method (like SHAP or LIME).
- **Data Privacy:** Never contribute datasets containing PII (Personally Identifiable Information). Use publicly available phishing feeds like PhishTank or OpenPhish.
- **Bias Mitigation:** Ensure the model does not unfairly flag legitimate international domains or specific TLDs without statistical justification.

---

##  Pull Request Process

1. **Fork** the repository and create your branch from `main`.
2. **Branch Naming:** Use descriptive names like `feat/new-lexical-feature` or `fix/gui-crash`.
3. **Testing:** Ensure the `step5_integrate_predict.py` (GUI) runs correctly with your changes.
4. **Documentation:** Update the `README.md` if your change adds new dependencies or changes the user interface.
5. **The PR Message:** Describe **what** you changed and **why**. Mention any related issues (e.g., `Closes #12`).

---

## Project Maintenance
This project is maintained by **[WASIK-S](https://github.com/WASIK-S)** and **[SalmaTech-03](https://github.com/SalmaTech-03)**. 

We review Pull Requests weekly. If your PR follows the standards above, it will likely be merged quickly!

**Thank you for helping us build a more secure AI!**
