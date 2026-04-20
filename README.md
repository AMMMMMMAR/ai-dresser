# Virtual AI Dressing (SmartFit) - Backend Repository

Welcome to the backend repository for our Virtual AI Dressing project!

This repository contains the core logic for our virtual ai fitting features. To keep our codebase clean and avoid merge conflicts as we work in parallel, we follow a strict branching workflow.

## 📌 Core Rules

1. **Never push directly to `main` or `dev`.** All work must be done on your own feature branch.

2. **Review Before Merging:** All Pull Requests (PRs) must be reviewed by at least one other team member before merging into `dev`.

## 🌳 Branching Strategy

We use a three-tier branch system:

- **`main`:** The production-ready branch. This is the polished, stable version of our project used for final testing and demonstrations.
- **`dev`:** The integration branch. This is where we combine all our individual features to test how they work together.
- **`feature/...`:** Your personal working branch.

## 🛠️ How to Work on a New Feature

Follow these exact steps when starting a new task:

**1. Update your local repository**
Always start by making sure you have the latest code from the `dev` branch:

```bash
git checkout dev
git pull origin dev
```
