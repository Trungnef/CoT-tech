import json

# Load the problems file
with open('db/questions/problems.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

total_problems = len(data['questions'])
print(f"Total problems: {total_problems}")

# Analyze problem types
types = {}
for q in data['questions']:
    types[q['type']] = types.get(q['type'], 0) + 1

print("\n=== Problem Types ===")
for t, count in sorted(types.items()):
    print(f"{t}: {count} ({count/total_problems*100:.1f}%)")

# Analyze difficulty levels
difficulty = {}
for q in data['questions']:
    difficulty[q['difficulty']] = difficulty.get(q['difficulty'], 0) + 1

print("\n=== Difficulty Levels ===")
for d, count in sorted(difficulty.items()):
    print(f"{d}: {count} ({count/total_problems*100:.1f}%)")

# Analyze tags
tags = {}
for q in data['questions']:
    for tag in q['tags']:
        tags[tag] = tags.get(tag, 0) + 1

print("\n=== Tags ===")
for t, count in sorted(tags.items()):
    print(f"{t}: {count} ({count/total_problems*100:.1f}%)") 