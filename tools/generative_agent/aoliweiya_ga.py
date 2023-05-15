from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from retrivers.llama_time_weighted_retriever import LlamaTimeWeightedVectorStoreRetriever
from vectorestores.chroma import EnhancedChroma
from generative_agents.llama_generative_agent import LlamaGenerativeAgent
from generative_agents.llama_memory import LlamaGenerativeAgentMemory


local_path = "../../models/game_npc_vicuna_huntress/ggml-f16.bin"
print("load LlamaCpp model ggml-f16.bin")
llm = LlamaCpp(model_path=local_path, verbose=True, n_batch=256, temperature=0.3, n_ctx=2048, use_mmap=False, stop=["###"])

# ----------------------------------------------------------------------------------------------------------------------

def create_new_memory_retriever():
    print("Load LlamaCpp embeddings ggml-f16.bin")
    embeddings_model = LlamaCppEmbeddings(model_path=local_path)
    vs = EnhancedChroma(embedding_function=embeddings_model)
    return LlamaTimeWeightedVectorStoreRetriever(vectorstore=vs, other_score_keys=["importance"], k=14)

# ----------------------------------------------------------------------------------------------------------------------

print("Aoliweiya LlamaGenerativeAgentMemory")
aoliweiya_memory = LlamaGenerativeAgentMemory(
    llm=llm,
    memory_retriever=create_new_memory_retriever(),
    reflection_threshold=8, # we will give this a relatively low number to show how reflection works
    verbose=True,
)

# ----------------------------------------------------------------------------------------------------------------------

print("Create Aoliweiya LlamaGenerativeAgent")
aoliweiya = LlamaGenerativeAgent(
    name="奥莉薇娅",
    age=25,
    traits="坚强, 干练, 大姐姐", # You can add more persistent traits here
    status="导师", # When connected to a virtual world, we can have the characters update their status
    memory_retriever=create_new_memory_retriever(),
    llm=llm,
    memory=aoliweiya_memory,
    verbose=True,
)

# ----------------------------------------------------------------------------------------------------------------------

print(aoliweiya.get_summary(force_refresh=True))

# ----------------------------------------------------------------------------------------------------------------------

# We can add memories directly to the memory object
print("Add memories to Aoliweiya")
aoliweiya_observations = [
    "猎人队长通过试训",
    "雪莉加入队伍",
    "妮可前来报名，但是晕倒在帐篷外",
    "艾丽娜怀疑妮可的能力，不希望她加入猎人小队",
    "猎人队长正在招聘新队员",
    "城外发现三只四级龙兽",
    "雪莉希望出城扫荡龙兽",
]
for observation in aoliweiya_observations:
    aoliweiya_memory.add_memory(observation)

# ----------------------------------------------------------------------------------------------------------------------

# Now that Aoliweiya has 'memories', their self-summary is more descriptive, though still rudimentary.
# We will see how this summary updates after more observations to create a more rich description.
print(aoliweiya.get_summary(force_refresh=True))

# ----------------------------------------------------------------------------------------------------------------------

print(aoliweiya.generate_reaction("猎人队长申请出城扫荡龙兽"))

# ----------------------------------------------------------------------------------------------------------------------

aoliweiya.generate_dialogue("领队大人", "你的小队准备好了吗？")

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
