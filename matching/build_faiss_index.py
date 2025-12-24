import os
import numpy as np
import faiss
from pymongo import MongoClient
import pickle

# MongoDB 연결 설정
MONGO_URI = "mongodb+srv://sth0824:daniel0824@sthcluster.sisvx.mongodb.net/?retryWrites=true&w=majority&appName=STHCluster"
client = MongoClient(MONGO_URI)
db = client["furniture_db"]
collection = db["furniture_embeddings"]

# FAISS 저장 경로
FAISS_INDEX_PATH = "faiss_index.faiss"
DOC_MAPPING_PATH = "doc_mapping.pkl"

def build_faiss_index():
    """FAISS 인덱스를 MongoDB 데이터로부터 생성하고 저장"""
    print("🔄 FAISS 인덱스를 빌드하는 중...")

    docs = list(collection.find({}))
    embeddings = []
    id_to_doc = {}

    for idx, doc in enumerate(docs):
        combined_vec = []
        if "cnn_embedding" in doc:
            combined_vec.extend(doc["cnn_embedding"])
        if "vit_embedding" in doc:
            combined_vec.extend(doc["vit_embedding"])
        if "clip_embedding" in doc:
            combined_vec.extend(doc["clip_embedding"])
        if "texture_embedding" in doc:
            combined_vec.extend(doc["texture_embedding"])

        if combined_vec:
            embeddings.append(np.array(combined_vec, dtype=np.float32))
            id_to_doc[idx] = doc  # FAISS 인덱스 ID와 문서 매핑

    if not embeddings:
        print("❌ 저장할 벡터가 없습니다!")
        return

    embeddings = np.array(embeddings)

    # FAISS L2 거리 기반 인덱스 생성 및 저장
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

    # FAISS 인덱스 저장
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    print(f"✅ FAISS 인덱스 저장 완료: {FAISS_INDEX_PATH}")

    # ID-문서 매핑 저장
    with open(DOC_MAPPING_PATH, "wb") as f:
        pickle.dump(id_to_doc, f)
    print(f"✅ 문서 매핑 저장 완료: {DOC_MAPPING_PATH}")


if __name__ == "__main__":
    build_faiss_index()
