# Constants for VLite application
class Constants:
    # Device options
    DEVICE_CPU = 'cpu'
    DEVICE_CUDA = 'cuda'
    DEVICE_MPS = 'mps'
    
    # Model details
    DEFAULT_MODEL = 'mixedbread-ai/mxbai-embed-large-v1'
    
    # Metadata keys
    METADATA_TEXT = 'text'
    METADATA_BINARY_VECTOR = 'binary_vector'
    METADATA_METADATA = 'metadata'
    
    # Precision types
    PRECISION_BINARY = 'binary'
    
    # Logging
    LOG_INIT = "[VLite.__init__] Initializing VLite with device: {}"
    LOG_EXEC_TIME = "[VLite.{}] Execution time: {:.5f} seconds"
    LOG_NO_COLLECTION = "[VLite.__init__] Collection file {} not found. Initializing empty attributes."
    LOG_RETRIEVING = "[VLite.retrieve] Retrieving similar texts..."
    LOG_RETRIEVING_QUERY = "[VLite.retrieve] Retrieving top {} similar texts for query: {}"
    LOG_RETRIEVAL_COMPLETED = "[VLite.retrieve] Retrieval completed."
    LOG_RANK_FILTER = "[VLite.rank_and_filter] Shape of query vector: {}"
    LOG_RANK_FILTER_RESHAPE = "[VLite.rank_and_filter] Shape of query vector after reshaping: {}"
    LOG_RANK_FILTER_CORPUS_SHAPE = "[VLite.rank_and_filter] Shape of corpus binary vectors array: {}"
    LOG_RANK_FILTER_TOP_K = "[VLite.rank_and_filter] Top {} indices: {}"
    LOG_RANK_FILTER_TOP_K_SCORES = "[VLite.rank_and_filter] Top {} scores: {}"
    LOG_RANK_FILTER_COLLECTION_COUNT = "[VLite.rank_and_filter] No. of items in the collection: {}"
    LOG_RANK_FILTER_VLITE_COUNT = "[VLite.rank_and_filter] Vlite count: {}"
    LOG_ADD_ENCODING = "[VLite.add] Encoding text... not chunking"
    
    TELEMETRY_POSTHOG = 'phc_i9Aq4aTt4aFpFqyKzxN9LGq3SfoIjYNAcazDnn6dSLP'
    TELEMETRY_POSTHOG_HOST = 'https://us.i.posthog.com'
