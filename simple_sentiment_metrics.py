import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re

def calculate_sentiment_metrics(file_path, true_col, pred_col):
    """
    Calculate and visualize sentiment analysis metrics from a CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file
        true_col (str): Column name containing true sentiment labels
        pred_col (str): Column name containing predicted sentiment labels
    """
    print(f"Loading data from {file_path}...")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with {len(df)} rows")
        
        # Check if columns exist
        if true_col not in df.columns:
            print(f"Error: Column '{true_col}' not found in the CSV file")
            print(f"Available columns: {', '.join(df.columns)}")
            return
            
        if pred_col not in df.columns:
            print(f"Error: Column '{pred_col}' not found in the CSV file")
            print(f"Available columns: {', '.join(df.columns)}")
            return
        
        # Extract sentiment labels from text
        def extract_sentiment(text):
            """Extract sentiment label from text format."""
            if not isinstance(text, str):
                return "Neutral"
                
            # Check for troll comments
            if "(TROLL)" in text:
                return "TROLL"
                
            # Extract basic sentiment
            match = re.match(r"(Positive|Negative|Neutral)", text)
            if match:
                return match.group(1)
            return "Neutral"  # Default
        
        # Process the data
        true_labels = df[true_col].apply(extract_sentiment).tolist()
        pred_labels = df[pred_col].apply(extract_sentiment).tolist()
        
        print("\nDistribution of true sentiment labels:")
        true_counts = pd.Series(true_labels).value_counts()
        for label, count in true_counts.items():
            print(f"  {label}: {count} ({count/len(true_labels)*100:.1f}%)")
        
        print("\nDistribution of predicted sentiment labels:")
        pred_counts = pd.Series(pred_labels).value_counts()
        for label, count in pred_counts.items():
            print(f"  {label}: {count} ({count/len(pred_labels)*100:.1f}%)")
        
        # Create label mappings for scikit-learn metrics
        unique_labels = sorted(list(set(true_labels + pred_labels)))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        
        # Convert to numeric labels
        y_true = np.array([label_to_id[label] for label in true_labels])
        y_pred = np.array([label_to_id[label] for label in pred_labels])
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate precision and recall
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Print overall metrics
        print("\n===== SENTIMENT ANALYSIS METRICS =====")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (macro): {precision:.4f}")
        print(f"Recall (macro): {recall:.4f}")
        print(f"F1 Score (macro): {f1:.4f}")
        
        # Calculate and print per-class metrics
        print("\n===== PER-CLASS METRICS =====")
        cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
        
        for i, label in enumerate(unique_labels):
            true_positives = cm[i, i]
            false_positives = sum(cm[:, i]) - true_positives
            false_negatives = sum(cm[i, :]) - true_positives
            
            class_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            class_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
            
            support = sum(cm[i, :])
            
            print(f"\n{label}:")
            print(f"  Precision: {class_precision:.4f}")
            print(f"  Recall: {class_recall:.4f}")
            print(f"  F1 Score: {class_f1:.4f}")
            print(f"  Support: {support} samples")
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        
        # Plot metrics bar chart
        plt.figure(figsize=(10, 6))
        metrics_dict = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
        
        # Create bar chart
        bars = plt.bar(
            metrics_dict.keys(), 
            metrics_dict.values(), 
            color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )
        
        # Add labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.01,
                f'{height:.4f}', 
                ha='center', 
                va='bottom'
            )
        
        plt.ylim(0, 1.1)
        plt.title('Sentiment Analysis Performance Metrics')
        plt.savefig("metrics_comparison.png")
        
        # Plot per-class metrics
        class_metrics = []
        for i, label in enumerate(unique_labels):
            true_positives = cm[i, i]
            false_positives = sum(cm[:, i]) - true_positives
            false_negatives = sum(cm[i, :]) - true_positives
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics.append({
                'Class': label,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            })
        
        # Convert to DataFrame for easier plotting
        class_df = pd.DataFrame(class_metrics)
        class_df = pd.melt(class_df, id_vars='Class', var_name='Metric', value_name='Score')
        
        # Create figure
        plt.figure(figsize=(12, 7))
        sns.barplot(x='Class', y='Score', hue='Metric', data=class_df)
        plt.title('Per-Class Performance Metrics')
        plt.ylim(0, 1.1)
        plt.legend(title='Metric')
        plt.tight_layout()
        plt.savefig("per_class_metrics.png")
        
        print("\nVisualizations saved to:")
        print("- confusion_matrix.png")
        print("- metrics_comparison.png")
        print("- per_class_metrics.png")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    # Ask for file path and column names
    file_path = input("Enter CSV file path (e.g., product_comments.csv): ")
    print("\nAvailable columns in the file:")
    
    try:
        df = pd.read_csv(file_path)
        for col in df.columns:
            print(f"- {col}")
        
        true_col = input("\nEnter column name with true sentiment labels: ")
        pred_col = input("Enter column name with predicted sentiment labels: ")
        
        calculate_sentiment_metrics(file_path, true_col, pred_col)
    except Exception as e:
        print(f"Error: {str(e)}")