import os
import sys

# Add the current directory to sys.path to ensure modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from skills.business_analyst.agent import BusinessAnalystAgent

def main():
    print("🚀 Starting AI Financial Agent System (Local Llama 3)...")
    print("-------------------------------------------------------")

    # Initialize Business Analyst
    # Adjust paths relative to main.py
    analyst = BusinessAnalystAgent(data_path="./data", db_path="./storage/chroma_db")

    while True:
        print("\nMenu:")
        print("1. Ingest Data (Process PDFs in /data)")
        print("2. Ask a Question")
        print("q. Quit")

        choice = input("Enter choice: ").lower()

        if choice == '1':
            analyst.ingest_data()
        elif choice == '2':
            query = input("\n👨‍💼 Ask the Business Analyst: ")
            result = analyst.analyze(query)
            print("\n📝 [Analyst Report]:")
            print(result)
        elif choice == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
