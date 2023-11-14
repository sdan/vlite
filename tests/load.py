from vlite.main import VLite

if __name__ == '__main__':
    option = input("Enter 1 to query a database and 2 to view all entries in the database: ")
    if int(option) == 1:
        location = input("Enter the location of the data file: ")
        db = VLite(location)
        query = input("Enter a query: ")
        data, metadata, score = db.remember(query, DEBUG=True)
        for d,m,s in zip(data, metadata, score):
            print("Data: ", d)
            print("Metadata: ", m)
            print("Score: ", s)
            print()
    elif int(option) == 2:
        location = input("Enter the location of the data file: ")
        db = VLite(location)
        data = db._data
        data_keys = list(data.keys())
        metadata = db._metadata
        metadata_keys = list(metadata.keys())
        vectors = db._vectors
        vector_key_store = db._vector_key_store

    inc = 0
    for data_key, metadata_key, vector_key in zip(data_keys, metadata_keys, vector_key_store):
        print("Entry ", inc)
        print("Data Key: ", data_key)
        print("Metadata Key: ", metadata_key)
        print("Vector Key: ", vector_key)
        print()
        inc += 1
    
    print("Total data entries: ", len(data_keys))
    print("Total metadata entries: ", len(metadata_keys))
    print("Total vector entries: ", len(vector_key_store))
    print("Total vector entries: ", len(vectors))
        