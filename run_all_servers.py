import subprocess
import sys
import os

def run_servers():
    # Activate virtual environment
    venv_path = os.path.join(os.getcwd(), 'venv', 'Scripts', 'activate.bat')
    if not os.path.exists(venv_path):
        print("Virtual environment not found. Please create it first.")
        sys.exit(1)

    # Command to activate venv and run servers
    activate_cmd = f'call "{venv_path}"'

    # Run RAG server (port 8000)
    rag_cmd = f'{activate_cmd} && python server2/server5.py --host localhost --port 8000'

    # Run no-RAG server (port 8001)
    no_rag_cmd = f'{activate_cmd} && python basic.py --host localhost --port 8001'

    print("Starting RAG server on port 8000...")
    rag_process = subprocess.Popen(rag_cmd, shell=True, cwd=os.getcwd())

    print("Starting no-RAG server on port 8001...")
    no_rag_process = subprocess.Popen(no_rag_cmd, shell=True, cwd=os.getcwd())

    print("Both servers are running. Press Ctrl+C to stop.")

    try:
        # Wait for both processes
        rag_process.wait()
        no_rag_process.wait()
    except KeyboardInterrupt:
        print("\nStopping servers...")
        rag_process.terminate()
        no_rag_process.terminate()
        rag_process.wait()
        no_rag_process.wait()
        print("Servers stopped.")

if __name__ == "__main__":
    run_servers()
