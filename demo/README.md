# 🎬 Demo Files

This folder contains everything you need to test the Agentic RAG system immediately.

## 📄 Files

| File | Purpose |
|------|---------|
| `company_handbook.pdf` | Sample enterprise document (8 pages, 6 policies) |
| `test_queries.json` | 12 test questions with expected topics |
| `quick_test.py` | Automated end-to-end test script |
| `test_results.json` | Generated after running quick_test.py |

## 🚀 Quick Start

### Option 1: Automated Test (Recommended)
```bash
# From project root
python demo/quick_test.py
```

This will:
1. ✅ Check your API key configuration
2. ✅ Ingest the demo PDF into ChromaDB
3. ✅ Run 5 test queries through both Agentic and Traditional RAG
4. ✅ Show side-by-side comparison
5. ✅ Generate performance summary
6. 💾 Save results to `demo/test_results.json`

### Option 2: Manual Testing
```python
from ingestion.pipeline import ingest_document
from pipeline import AgenticRAGPipeline

# Ingest demo document
ingest_document("demo/company_handbook.pdf")

# Ask questions
pipeline = AgenticRAGPipeline()
result = pipeline.run("What is the remote work policy?")
print(result["final_answer"])
```

### Option 3: Streamlit UI
```bash
streamlit run frontend/app.py
# Then upload demo/company_handbook.pdf through the UI
```

## 📋 Test Queries

The `test_queries.json` contains 12 questions covering:

| # | Query | Category | Difficulty |
|---|-------|----------|------------|
| 1 | What is the remote work policy? | Factual | Simple |
| 2 | How many vacation days per year? | Factual | Simple |
| 3 | Procedure for requesting time off? | Procedural | Medium |
| 4 | Compare full-time vs part-time PTO? | Comparative | Medium |
| 5 | Security requirements for confidential data? | Factual | Medium |
| 6 | Explain the code of conduct? | Procedural | Medium |
| 7 | Reimbursement policy for business travel? | Factual | Medium |
| 8 | How does performance review work? | Procedural | Complex |
| 9 | IT support contact details? | Factual | Simple |
| 10 | Describe the onboarding process? | Procedural | Complex |
| 11 | Per diem rate for domestic travel? | Factual | Simple |
| 12 | Can employees work remotely every day? | Factual | Simple |

## 🏢 Demo Document Content

The `company_handbook.pdf` contains realistic enterprise policies:

1. **Remote Work Policy** - 3 days/week, manager approval, eligibility criteria
2. **PTO Policy** - 20 days/year, rollover rules, holiday schedule
3. **Security Policy** - Passwords, MFA, data classification, breach reporting
4. **Performance Reviews** - Biannual reviews, 360 feedback, dimensions
5. **Travel Reimbursement** - Pre-approval, limits, per diem rates
6. **Code of Conduct** - Ethics, reporting, no-retaliation policy
7. **IT Support** - Contact info, hours, common issues
8. **Onboarding** - 2-week program, 30-60-90 day plan

## 🎯 What to Look For

When running the comparison, observe:

1. **Agentic RAG** will show:
   - Query understanding (intent detection)
   - Routing decisions (strategy selection)
   - Query rewriting (acronym expansion)
   - Validation (filtering irrelevant chunks)
   - Reflection (self-correction loops)
   - Better citations with source pages

2. **Traditional RAG** will show:
   - Direct retrieval without optimization
   - No validation of context quality
   - No self-correction
   - Generic answers without citations

## 📊 Expected Results

After running `quick_test.py`, you should see:

```
Agentic RAG:    ~15-25s per query | 3-5 validated docs | 0-1 iterations
Traditional RAG: ~5-10s per query | 5 docs retrieved | no validation
```

The Agentic system takes longer but produces higher-quality, cited answers.

## 🔧 Troubleshooting

**"No documents in DB"**
- Run ingestion first: `python -c "from ingestion.pipeline import ingest_document; ingest_document('demo/company_handbook.pdf')"`

**"API key error"**
- Check `.env` file has `GROQ_API_KEY=your_key_here`
- Get free key at: https://console.groq.com/

**"Module not found"**
- Install requirements: `pip install -r requirements.txt`
- Ensure you're in the project root directory
