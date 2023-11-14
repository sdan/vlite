from vlite.main import VLite

if __name__ == '__main__':
    db = VLite()
    db.memorize("Tadpoles love to eat algae and small insects such as mosquito larvae.", id="One")
    db.memorize("Cats are the most popular pet in the United States.", metadata={"type": "animal"})
    db.memorize("The Amazon rainforest is the largest rainforest in the world.", id="Two")
    db.memorize("The first person on the moon was Neil Armstrong.", metadata={"type": "person"})
    db.memorize("The Declaration of Independence was signed in 1776.", id="Three")
    db.memorize("The capital of the United States is Washington, D.C.", metadata={"type": "place"})
    db.memorize("The largest country in the world by area is Russia.", id="Four")
    db.memorize("Dogs are descended from wolves.", metadata={"type": "animal"})
    db.memorize("The longest river in the world is the Nile.", id="Five")
    db.memorize("American football is the most popular sport in the United States.", metadata={"type": "sport"})
    print(f"{db.entry_count} entries added to database.")

    data, metadata, score = db.remember("What do tadpoles eat?")
    print("Data: ", data)
    print("Metadata: ", metadata)
    print("Score: ", score)

    db.forget("One")
    db.forget("Two")
    db.forget("Three")
    db.forget("Four")
    db.forget("Five")
    print(f"{db.entry_count} entries remaining in database.\n")

    print("Keys:")
    for vector in db._vector_key_store:
        print(vector)