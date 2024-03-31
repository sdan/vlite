from random_word import RandomWords
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def generate_string_of_length(target_tokens: int) -> str:
    r = RandomWords()
    generated_string = ""
    current_tokens = 0

    while current_tokens < target_tokens:
        word = r.get_random_word()
        word_tokens = num_tokens_from_string(word)

        if current_tokens + word_tokens <= target_tokens:
            generated_string += word + " "
            current_tokens += word_tokens + 1  # Add 1 for the space
        else:
            break

    # Remove the trailing space
    generated_string = generated_string.strip()

    # If the token count is less than the target, append words one by one
    while current_tokens < target_tokens:
        word = r.get_random_word()
        word_tokens = num_tokens_from_string(word)

        if current_tokens + word_tokens <= target_tokens:
            generated_string += " " + word
            current_tokens += word_tokens + 1  # Add 1 for the space
        else:
            break

    # If the token count is greater than the target, remove words one by one
    while current_tokens > target_tokens:
        words = generated_string.split()
        last_word = words.pop()
        last_word_tokens = num_tokens_from_string(last_word)
        current_tokens -= last_word_tokens + 1  # Subtract 1 for the space
        generated_string = " ".join(words)

    return generated_string

# Generate a string of 512 tokens
string_512_tokens = generate_string_of_length(512)
print(f"String of 512 tokens:\n{string_512_tokens}")
print(f"Actual token count: {num_tokens_from_string(string_512_tokens)}")

print("\n" + "-" * 50 + "\n")

# Generate a string of 8192 tokens
string_8192_tokens = generate_string_of_length(8192)
print(f"String of 8192 tokens:\n{string_8192_tokens}")
print(f"Actual token count: {num_tokens_from_string(string_8192_tokens)}")