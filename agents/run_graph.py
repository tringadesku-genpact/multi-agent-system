from agents.graph import run


def main():
    task = input("Enter task: ").strip()
    if not task:
        return

    out = run(task=task, top_k=5)

    print("\n=== DRAFT ===\n")
    print(out.get("draft", ""))

    print("\n=== TRACE ===")
    for e in out.get("trace", []):
        print(f"- {e['agent']} :: {e['action']} :: {e.get('detail','')}")


if __name__ == "__main__":
    main()
