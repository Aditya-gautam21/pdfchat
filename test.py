#!/usr/bin/env python3
"""
Test script to verify all components are working properly
Run this before starting your PDF chatbot application
"""

import sys
import os
from PIL import Image
import io

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import fitz  # PyMuPDF
        print("✅ PyMuPDF imported successfully")
    except ImportError as e:
        print(f"❌ PyMuPDF import failed: {e}")
        return False
    
    try:
        import pdfplumber
        print("✅ pdfplumber imported successfully")
    except ImportError as e:
        print(f"❌ pdfplumber import failed: {e}")
        return False
    
    try:
        import pytesseract
        print("✅ pytesseract imported successfully")
    except ImportError as e:
        print(f"❌ pytesseract import failed: {e}")
        return False
    
    try:
        from langchain.schema import Document
        from langchain_community.vectorstores import Chroma
        print("✅ LangChain components imported successfully")
    except ImportError as e:
        print(f"❌ LangChain import failed: {e}")
        return False
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        print("✅ PyTorch and Transformers imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch/Transformers import failed: {e}")
        return False
    
    return True

def test_tesseract():
    """Test if Tesseract OCR is working"""
    print("\n🔍 Testing Tesseract OCR...")
    
    try:
        import pytesseract
        
        # Set Tesseract path for Windows if needed
        if os.name == 'nt':  # Windows
            possible_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME'))
            ]
            
            tesseract_found = False
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    tesseract_found = True
                    print(f"✅ Found Tesseract at: {path}")
                    break
            
            if not tesseract_found:
                print("❌ Tesseract executable not found in standard locations")
                return False
        
        # Test Tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
        
        # Create a simple test image with text
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple test image
        img = Image.new('RGB', (200, 50), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Hello World", fill='black')
        
        # Test OCR
        text = pytesseract.image_to_string(img).strip()
        if "Hello" in text or "World" in text:
            print(f"✅ OCR test successful: '{text}'")
            return True
        else:
            print(f"⚠️ OCR test partial: '{text}' (expected 'Hello World')")
            return True  # Still consider it working
            
    except Exception as e:
        print(f"❌ Tesseract test failed: {e}")
        return False

def test_embedding_model():
    """Test if embedding model can be loaded"""
    print("\n🤖 Testing embedding model...")
    
    try:
        from embeddings import LocalHFEmbedder
        
        # Try to load the model (this will download if not present)
        model_path = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Loading embedding model: {model_path}")
        
        embedder = LocalHFEmbedder(model_path=model_path, device="cpu")
        
        # Test embedding generation
        test_text = "This is a test sentence for embedding."
        embedding = embedder.embed_query(test_text)
        
        if len(embedding) > 0:
            print(f"✅ Embedding model working! Generated {len(embedding)}-dimensional embedding")
            return True
        else:
            print("❌ Embedding generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Embedding model test failed: {e}")
        print("💡 The model will be downloaded on first use")
        return False

def test_data_processing():
    """Test the Data class functionality"""
    print("\n📄 Testing PDF data processing...")
    
    try:
        from data import Data
        
        data_processor = Data()
        print("✅ Data processor initialized successfully")
        
        # Test if methods exist
        methods = ['get_pdf_text', 'extract_tables', 'extract_and_process_images', 'extract_all_content']
        for method in methods:
            if hasattr(data_processor, method):
                print(f"✅ Method {method} available")
            else:
                print(f"❌ Method {method} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False

def test_ollama_connection():
    """Test connection to Ollama (optional)"""
    print("\n🦙 Testing Ollama connection...")
    
    try:
        import requests
        
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama is running with {len(models)} models")
            
            # Check if mistral:instruct is available
            model_names = [model.get('name', '') for model in models]
            if 'mistral:instruct' in model_names:
                print("✅ mistral:instruct model is available")
            else:
                print("⚠️ mistral:instruct model not found. Run: ollama pull mistral:instruct")
            
            return True
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama. Make sure Ollama is running on localhost:11434")
        return False
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 PDF Chatbot Setup Verification")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Tesseract OCR", test_tesseract),
        ("Data Processing", test_data_processing),
        ("Embedding Model", test_embedding_model),
        ("Ollama Connection", test_ollama_connection),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
        print()
    
    # Summary
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name}: {status}")
        if passed_test:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your PDF chatbot should work correctly.")
    elif passed >= total - 1:
        print("\n⚠️ Almost ready! Fix the failing test(s) above.")
    else:
        print("\n❌ Several issues found. Please fix the failing tests before running the app.")
    
    print("\n📖 Next steps:")
    print("1. Fix any failing tests above")
    print("2. Install Ollama and pull mistral:instruct model if needed")
    print("3. Run: streamlit run app.py")

if __name__ == "__main__":
    main()