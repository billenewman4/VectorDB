#!/usr/bin/env python3
"""
generate_management_email.py - Create a management summary email from LLM cluster analyses

This script reads the detailed LLM analyses of product clusters and generates a concise
email summary highlighting key issues and recommendations for management review.
"""

import os
import re
import glob
import json
import argparse
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_cluster_analyses(analysis_dir: str) -> List[Dict[str, Any]]:
    """
    Load all cluster analysis files from the specified directory.
    
    Args:
        analysis_dir: Directory containing cluster analysis files
        
    Returns:
        List of cluster analysis data dictionaries
    """
    analyses = []
    
    # Get all markdown analysis files
    analysis_files = glob.glob(os.path.join(analysis_dir, "cluster_cluster_*_analysis.md"))
    
    for file_path in analysis_files:
        # Extract cluster ID from filename
        cluster_id = re.search(r'cluster_cluster_(\d+)_analysis', file_path).group(1)
        
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Try to extract key information using regex
        stats_match = re.search(r'## Cluster Statistics\s+(.+?)##', content, re.DOTALL)
        products_match = re.search(r'## Products in Cluster\s+(.+?)##', content, re.DOTALL)
        analysis_match = re.search(r'## LLM Analysis\s+(.+)', content, re.DOTALL)
        
        stats = stats_match.group(1).strip() if stats_match else ""
        products = products_match.group(1).strip() if products_match else ""
        analysis = analysis_match.group(1).strip() if analysis_match else ""
        
        # Extract product descriptions
        product_descriptions = []
        if products:
            # Look for descriptions in the table
            desc_matches = re.findall(r'\|\s+\d+\s+\|\s+(.+?)\s+\|', products)
            if desc_matches:
                product_descriptions = desc_matches
        
        analyses.append({
            "cluster_id": cluster_id,
            "stats": stats,
            "products": product_descriptions,
            "analysis": analysis,
            "file_path": file_path
        })
    
    return analyses

def extract_key_insights(analysis_text: str) -> Dict[str, str]:
    """
    Extract key insights from the analysis text.
    
    Args:
        analysis_text: The full analysis text
        
    Returns:
        Dictionary with problem, reasons, and recommendations
    """
    problem = ""
    reasons = ""
    recommendations = ""
    opportunities = ""
    
    # Try to extract problem statement
    problem_match = re.search(r'#### 1\. Inconsistencies.+?(\n.+?)+?\n\n', analysis_text, re.DOTALL)
    if problem_match:
        problem = problem_match.group(0).strip()
    
    # Try to extract reasons
    reasons_match = re.search(r'#### 2\. Potential Reasons.+?(\n.+?)+?\n\n', analysis_text, re.DOTALL)
    if reasons_match:
        reasons = reasons_match.group(0).strip()
    
    # Try to extract recommendations
    recommendations_match = re.search(r'#### 3\. Recommendations.+?(\n.+?)+?\n\n', analysis_text, re.DOTALL)
    if recommendations_match:
        recommendations = recommendations_match.group(0).strip()
    
    # Try to extract opportunities
    opportunities_match = re.search(r'#### 4\. Opportunity.+?(\n.+?)+?\n\n', analysis_text, re.DOTALL)
    if opportunities_match:
        opportunities = opportunities_match.group(0).strip()
    
    return {
        "problem": problem,
        "reasons": reasons,
        "recommendations": recommendations,
        "opportunities": opportunities
    }

def generate_email_with_llm(analyses: List[Dict[str, Any]], llm_model: str = "gpt-4o") -> str:
    """
    Generate a management email summary using an LLM.
    
    Args:
        analyses: List of cluster analyses
        llm_model: LLM model to use
        
    Returns:
        Formatted email text
    """
    # Initialize LLM
    llm = ChatOpenAI(model=llm_model, temperature=0.3)
    
    # Prepare data for the LLM
    clusters_data = []
    for analysis in analyses:
        # Get sample product descriptions (up to 3)
        sample_products = analysis["products"][:3]
        sample_products_text = ", ".join(sample_products) if sample_products else "Unknown products"
        if len(analysis["products"]) > 3:
            sample_products_text += ", ..."
        
        # Extract key insights
        insights = extract_key_insights(analysis["analysis"])
        
        # Add to clusters data
        clusters_data.append({
            "cluster_id": analysis["cluster_id"],
            "products": sample_products_text,
            "problem": insights["problem"],
            "opportunities": insights["opportunities"]
        })
    
    # Convert to a string representation for the prompt
    clusters_text = ""
    for i, cluster in enumerate(clusters_data, 1):
        clusters_text += f"Cluster {i}: ID {cluster['cluster_id']}\n"
        clusters_text += f"Products: {cluster['products']}\n"
        clusters_text += f"Problem:\n{cluster['problem']}\n"
        clusters_text += f"Opportunities:\n{cluster['opportunities']}\n\n"
    
    # Create the prompt
    template = """You are the head of pricing analytics at a retail company. 
    
    Write a concise, professional email to the Director of Merchandising summarizing the key findings from our product pricing analysis. 
    The email should highlight the most important issues discovered in our product clusters and provide actionable recommendations.
    
    Here are the detailed findings from our analysis:
    
    {clusters_data}
    
    Your email should:
    1. Have a clear, attention-grabbing subject line
    2. Include a brief introduction explaining the purpose of the analysis
    3. Highlight 3-5 key findings across the clusters in a concise, bulleted format
    4. For each finding, include: Cluster ID, Product Category, Problem, and Recommended Action
    5. Conclude with 2-3 high-level recommendations and next steps
    6. Use a professional but conversational tone
    7. Keep the entire email concise (no more than 2-3 paragraphs plus bullets)
    
    Format the email with proper sections (To, From, Subject, Body) as if it were being sent today.
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["clusters_data"]
    )
    
    chain = prompt | llm | StrOutputParser()
    
    # Generate the email
    return chain.invoke({"clusters_data": clusters_text})

def main():
    parser = argparse.ArgumentParser(description='Generate management email from cluster analyses')
    parser.add_argument('--analysis_dir', type=str, default='product_clustering/data/llm_analysis',
                        help='Directory containing cluster analysis files')
    parser.add_argument('--output_dir', type=str, default='product_clustering/data/management_email',
                        help='Directory to save email file')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='LLM model to use')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load cluster analyses
    print("Loading cluster analyses...")
    analyses = load_cluster_analyses(args.analysis_dir)
    print(f"Found {len(analyses)} cluster analyses")
    
    # Generate email
    print(f"Generating management email using {args.model}...")
    email_text = generate_email_with_llm(analyses, args.model)
    
    # Save email to file
    today = datetime.now().strftime("%Y-%m-%d")
    email_path = os.path.join(args.output_dir, f"pricing_analysis_summary_{today}.md")
    with open(email_path, 'w') as f:
        f.write(email_text)
    
    print(f"Email generated and saved to {email_path}")

if __name__ == "__main__":
    main()
