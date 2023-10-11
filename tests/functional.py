from vlite.main import VLite

if __name__ == '__main__':
    db = VLite()
    db.memorize("Tadpoles love to eat algae and small insects such as mosquito larvae.")
    db.memorize("Cats are the most popular pet in the United States.")
    db.memorize("The Amazon rainforest is the largest rainforest in the world.")
    db.memorize("The first person on the moon was Neil Armstrong.")
    db.memorize("The Declaration of Independence was signed in 1776.")
    db.memorize("The capital of the United States is Washington, D.C.")
    db.memorize("The largest country in the world by area is Russia.")
    db.memorize("Dogs are descended from wolves.")
    db.memorize("The longest river in the world is the Nile.")
    db.memorize("American football is the most popular sport in the United States.")

    data, metadata, score = db.remember("What do tadpoles eat?")
    print("Data: ", data)
    print("Metadata: ", metadata)
    print("Score: ", score)