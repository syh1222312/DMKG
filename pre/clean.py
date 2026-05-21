import json
def stream_json_array(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # Skip leading whitespace and expect [
        while True:
            char = f.read(1)
            if not char:
                raise EOFError("Unexpected end of file")
            if char.isspace():
                continue
            if char == '[':
                break
            else:
                raise ValueError("File does not start with '['")
        while True:
            # Skip whitespace after [ or ,
            while True:
                char = f.read(1)
                if not char:
                    raise EOFError("Unexpected end of this file")
                if not char.isspace():
                    break
            if char == ']':
                return  # End of array
            if char != '{':
                raise ValueError("Expected '{' for object start")
            buffer = char
            brace_count = 1
            in_string = False
            escape = False
            while brace_count > 0:
                char = f.read(1)
                if not char:
                    raise EOFError("Unexpected end of file")
                buffer += char
                if in_string:
                    if escape:
                        escape = False
                    elif char == '\\':
                        escape = True
                    elif char == '"':
                        in_string = False
                elif char == '"':
                    in_string = True
                elif char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
            # Parse the buffer as JSON object
            obj = json.loads(buffer)
            yield obj
            # Expect , or ]
            while True:
                char = f.read(1)
                if not char:
                    raise EOFError("Unexpected end of file")
                if char.isspace():
                    continue
                if char == ',':
                    break
                if char == ']':
                    return
                raise ValueError("Expected ',' or ']' after object")


input_file = '../INITDATASET/dblp-v12.json'
output_file = '../INITDATASET/dblp-v12-clean.json'

# First pass: collect all potential objects and count author articles
author_count = {}
candidate_objs = []
seen_ids = set()

for obj in stream_json_array(input_file):
    if 'id' not in obj:
        continue
    id_ = obj['id']
    if id_ in seen_ids:
        continue
    seen_ids.add(id_)
    if 'title' not in obj or len(obj['title'].split()) < 5:
        continue
    if 'authors' not in obj or len(obj['authors']) < 5:
        continue
    if 'references' not in obj or len(obj['references']) < 10:
        continue

    # Count authors (assuming authors are list of dicts with 'name' or 'id'; using 'id' if present, else 'name')
    for author in obj.get('authors', []):
        author_key = author.get('id', author.get('name'))  # Use 'id' if exists, else 'name'
        if author_key is not None:
            author_count[author_key] = author_count.get(author_key, 0) + 1

    candidate_objs.append(obj)

# Second pass: filter authors in each object and write output
with open(output_file, 'w', encoding='utf-8') as out:
    out.write('[\n')
    first = True
    for obj in candidate_objs:
        # Filter authors with at least 5 articles
        filtered_authors = []
        for author in obj.get('authors', []):
            author_key = author.get('id', author.get('name'))
            if author_key is not None and author_count.get(author_key, 0) >= 5:
                filtered_authors.append(author)

        if len(filtered_authors) < 5:
            continue  # Skip if after filtering, authors < 5

        obj['authors'] = filtered_authors  # Update authors

        if not first:
            out.write(',\n')
        first = False
        json.dump(obj, out)
    out.write('\n]')



