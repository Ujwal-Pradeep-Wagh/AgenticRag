# Troubleshooting Guide

## Common Issues and Solutions

### 1. ModuleNotFoundError: No module named 'langchain.schema'

**Issue:** Old cached Python bytecode using deprecated import paths.

**Solution:**
```bash
# Clear Python cache
Remove-Item -Path __pycache__ -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path agents/__pycache__ -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Filter "*.pyc" -Recurse | Remove-Item -Force

# Run again
python demo/quick_test.py
```

### 2. ModuleNotFoundError: No module named 'torchvision'

**Issue:** Streamlit file watcher examining transformers library which references torchvision.

**Impact:** Non-critical warning - app still works fine.

**Solution (Option 1 - Recommended):** Ignore the warning, it doesn't affect functionality.

**Solution (Option 2):** Run with file watcher disabled:
```bash
streamlit run frontend/app.py --server.fileWatcherType none
```

Or use the wrapper script:
```bash
python run_frontend.py
```

**Solution (Option 3):** Install torchvision (large package ~500MB):
```bash
pip install torchvision
```

### 3. Model not found: 'qwen3-32b'

**Issue:** The model name doesn't exist or you don't have access.

**Solution:** Update your `.env` file with a valid Groq model:
```env
# Valid Groq models (as of 2026):
LLM_MODEL=llama-3.3-70b-versatile
# or
LLM_MODEL=mixtral-8x7b-32768
# or
LLM_MODEL=gemma2-9b-it
```

Then check available models:
```bash
python -c "from langchain_groq import ChatGroq; print('Available models: Check https://console.groq.com/docs/models')"
```

### 4. GROQ_API_KEY not set

**Issue:** Missing or invalid API key.

**Solution:**
1. Get API key from: https://console.groq.com/keys
2. Update `.env` file:
```env
GROQ_API_KEY=gsk_your_actual_key_here
```

### 5. Pipeline returns 0 documents

**Issue:** Documents not ingested or vector database empty.

**Solution:**
```bash
# Verify ingestion
python -c "from ingestion.pipeline import DocumentIngestionPipeline; p = DocumentIngestionPipeline(); print(p.get_stats())"

# Re-ingest demo document
python demo/quick_test.py
```

### 6. ImportError: cannot import name 'X' from 'langchain'

**Issue:** Version mismatch between langchain packages.

**Solution:**
```bash
# Reinstall all langchain packages
pip uninstall -y langchain langchain-core langchain-groq langchain-community langchain-chroma langchain-text-splitters langchain-huggingface
pip install -r requirements.txt
```

### 7. ChromaDB persistence issues

**Issue:** Vector database not persisting or corrupted.

**Solution:**
```bash
# Clear and recreate
Remove-Item -Path ./vector_db -Recurse -Force -ErrorAction SilentlyContinue
python demo/quick_test.py
```

### 8. Windows symlink warnings with HuggingFace

**Issue:** HuggingFace cache can't create symlinks on Windows.

**Impact:** Slower downloads, more disk space used, but still works.

**Solution (Option 1):** Ignore - it still works fine.

**Solution (Option 2):** Enable Developer Mode in Windows:
- Settings → Update & Security → For developers → Developer Mode

**Solution (Option 3):** Run Python as Administrator (not recommended for security).

**Solution (Option 4):** Set environment variable to suppress warning:
```env
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

## Verification Commands

### Check Python environment
```bash
python --version  # Should be 3.10+
pip list | Select-String "langchain"
```

### Test imports
```bash
python -c "from pipeline import AgenticRAGPipeline; print('✅ Pipeline imports OK')"
python -c "from ingestion.pipeline import DocumentIngestionPipeline; print('✅ Ingestion imports OK')"
python -c "import streamlit; print('✅ Streamlit imports OK')"
```

### Test API connection
```bash
python -c "from langchain_groq import ChatGroq; from config import Config; llm = ChatGroq(api_key=Config.GROQ_API_KEY, model='llama-3.3-70b-versatile'); print(llm.invoke('Hello').content)"
```

### Check vector database
```bash
python -c "from ingestion.pipeline import DocumentIngestionPipeline; print(DocumentIngestionPipeline().get_stats())"
```

## Getting Help

If issues persist:
1. Check logs in console output
2. Verify all dependencies: `pip install -r requirements.txt`
3. Check `.env` file configuration
4. Review `demo/README.md` for usage examples
5. Run with verbose logging: `python demo/quick_test.py 2>&1 | Out-File -FilePath debug.log`

## Performance Tips

1. **Use GPU for embeddings** (if available):
   ```env
   EMBEDDING_DEVICE=cuda
   ```

2. **Adjust chunk size** for better retrieval:
   ```env
   CHUNK_SIZE=256  # Smaller for precise retrieval
   CHUNK_SIZE=1024 # Larger for more context
   ```

3. **Tune retrieval parameters**:
   ```env
   TOP_K_RETRIEVAL=10  # Retrieve more candidates
   ```

4. **Disable file watcher** for faster Streamlit:
   ```bash
   streamlit run frontend/app.py --server.fileWatcherType none
   ```
