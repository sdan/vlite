# Vlite VDB Python Project Roadmap

This document outlines the roadmap for the Vlite VDB (Vector Database), a Python project. The Vlite VDB is known for its lightweight and high-performance characteristics but currently lacks full CRUD (Create, Read, Update, Delete) support and has some areas that need improvement. This roadmap defines the features and changes planned to transform the VDB into a more complete and user-friendly module while adhering to its original vision.

## The Next Mile
 - ~~Fix "Remember by ID" functionality.~~ **(Done)**
    - ~~"Remember by ID" was implemented but was not functioning appropriately.~~
- ~~Remove Chunking~~ **(Done)**
    - ~~There is a function that automatically chunks and parses text/data by character count. However, we believe that preparsing should be done outside of the library and handled by the user, not the library.~~
    - ~~By removing this, there will also be some simplifying changes regarding how chunks are currently stored.~~
- Revisit Metadata and Vector Storage.
    - Currently the metadata and vectors are stored in a simple but odd way as a slap dash method for immediate support on a project.
    - The structure of the keys, vectors, data and metadata will be revisited in interest of simplifying the current code.
- General Longterm Support.
    - Some of the functions have lots of comments and debugging code.
    - Unit tests could be better and are quite useless.
    - There is currently quite a bit of undescriptive verbosity that needs to be cleaned up.
    - All of these will be cleaned up before adding new functionality.
- ~~Delete~~ **(DONE)**
    - ~~There is currently no support for deleting items in the VDB.~~
    - ~~There are several use cases, such as bad entries from automatic data parsing or items you no longer want available for query like discontinued items, in which deletion rather than remaking the database is preferred.~~
    - ~~For sure, you'll be able to delete by ID.~~
    - ~~It is possible delete by similarity might be a useful, but it is not clear what scenario that would be. So this is not immediately planned unless there seems to be a regular use case for this.~~
- Update
    - Currently, there is no support for updating any aspect of individual data entries.
    - There will be support for updating the data, the metadata, and the key.
    - The vectors will naturally have to be updated if the data is updated.
    - There will be definite support for key based updates.
    - It is very possible that there could be similarity based updates as it is clear that automated work flows would likely benefit from updating highly similar entries. However, it is easy enough to implement by yourself after an Update method exists so it may be a while after basic Update support is present.

## The Next Ten Miles
- Graphs
    - Graphs offer an amazing opportunity to pull relevant information that may not be highly semantically similar.
    - For example, in wikipedia, under the topic quantum computing, you may just want to know about programming a quantum computer. But, the wiki article has many details about quantum physics. Maybe initially you don't care, but if you require explanations of particular effects, it would be nice if VDBs could link related but semantically different topics.
    - Additonally, in the context of IT assistance, it is common for questions to frequently appear when a new user is onboarded or a change in the product or service happens. These conversations usually lead to a similar result even though the conversations may vary. In this context, making a graph that branches conversations can lead to faster resolution by following the shortest paths. Either AI or Augmented Human in the Loop services can use entries as reference.
- Multimodal Similarities
    - While many of the new LLMs claim to be "Multimodal", in reality this is not likely the case. They are likely to be several models working together in clever ways to provide a singular AI experience. Which is fine.
    - The biggest problem with this approach is that each model will have it's own embeddings and direct similarity comparison between a picture and text is not likely to yeild any useful results.
    - There is an opporunity to support complex similarity comparison by allowing the user to supply their own embeddings or high dimensional representations of many data types.

## Other Ideas
For the 1 and 10 mile roadmaps, this is definitely not an exhaustive list. These are some of the key areas we think are either immediately useful from existing project we are working on or promising ideas that could lead to further improvements in AI and/or vector database search. But, if you think there is functionality or research areas you would like to see explored or implemented please feel free to contact us.
