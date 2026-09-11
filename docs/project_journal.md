# PaymentGuard Project Journal

## September 2026

### Repository and Environment Setup

I created a separate Conda environment for PaymentGuard because, after completing my master's at TIFR, I spent time looking at how data-science projects are organized. I understood that different projects may require different packages and versions. A separate environment lets me install and verify the tools needed for PaymentGuard without affecting my other Python projects.

I had already used Anaconda, Jupyter, Python, GitHub, SourceTree, and Overleaf. However, I had not previously used SQL as a complete part of a project. For PaymentGuard, I installed and verified SQLite and SQLAlchemy support because SQL analysis will be an important component of the project.

Previously, I mainly used SourceTree for Git operations. During this setup, I began learning how to perform the same operations directly in Terminal: checking the repository status, staging selected files, creating commits, and pushing them to GitHub. I also learned that Terminal and SourceTree are two ways of working with the same Git repository. A clear commit history allows other people, including recruiters, to follow how the project developed. My goal is to become comfortable using Python, Git, GitHub, and SourceTree together in one reproducible workflow.

In my earlier work, I often kept most of the analysis inside Jupyter notebooks. For PaymentGuard, I wanted to understand how a complete data-science project is organized, so I created separate folders for different types of work. The `data` folder keeps the original and processed datasets separate, while `src` contains Python code that can be reused outside a notebook. I will keep SQL queries in `sql`, correctness checks in `tests`, and explanations of my decisions in `docs`. The `dashboard` folder will contain the final interactive application, and `results` will store selected tables and figures. This structure should make the project easier for me to understand, reproduce, test, and explain to someone else.
