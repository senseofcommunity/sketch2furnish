import os
import torch
import numpy as np
import faiss
import pickle
from PIL import Image
from fastapi import FastAPI, HTTPException, Query, Body
from pymongo import MongoClient
from torchvision import transforms
from transformers import ViTImageProcessor, ViTModel
from torchvision.models import resnet50, ResNet50_Weights
import clip
import uvicorn
from pydantic import BaseModel

app = FastAPI()

# FAISS 인덱스 & 문서 매핑 로드
FAISS_INDEX_PATH = "C:/Users/LG/elice/sketch2furnish/matching/faiss_index.faiss"
DOC_MAPPING_PATH = "C:/Users/LG/elice/sketch2furnish/matching/doc_mapping.pkl"
faiss_index = None
id_to_doc = {}

def load_faiss_index():
    """FAISS 인덱스 및 문서 매핑 로드"""
    global faiss_index, id_to_doc
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOC_MAPPING_PATH):
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(DOC_MAPPING_PATH, "rb") as f:
            id_to_doc = pickle.load(f)

        if faiss_index.ntotal == 0:
            print("❌ FAISS 인덱스가 비어 있습니다. 인덱스를 다시 구축하세요.")
        else:
            print(f"✅ FAISS 인덱스 로드 완료! 벡터 개수: {faiss_index.ntotal}")
    else:
        print("❌ FAISS 인덱스 파일이 없습니다. 'build_faiss_index.py' 실행 필요!")

# MongoDB 설정
MONGO_URI = "mongodb+srv://sth0824:daniel0824@sthcluster.sisvx.mongodb.net/?retryWrites=true&w=majority&appName=STHCluster"
client = MongoClient(MONGO_URI)
db = client["furniture_db"]
collection = db["furniture_embeddings"]


# 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CNN 모델 (ResNet50)
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval().to(device)

# ViT 모델
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
vit_model.eval()

# CLIP 모델
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# 이미지 전처리
cnn_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 요청 데이터 모델
class RecommendationRequest(BaseModel):
    generated_img_path: str
    min_price: int
    max_price: int
    top_k: int = 4

@app.get("/")
async def read_root():
    """서버 상태 확인"""
    return {"message": "server is running"}

def search_faiss(query_vector, top_k=10):
    """FAISS 인덱스를 이용한 유사도 검색"""
    if faiss_index is None or faiss_index.ntotal == 0:
        print("❌ FAISS 인덱스가 로드되지 않음")
        return []

    query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
    distances, indices = faiss_index.search(query_vector, top_k)
    
    return [id_to_doc[idx] for idx in indices[0] if idx in id_to_doc]

def get_recommendations(generated_img_path: str, min_price: int, max_price: int, top_k: int):
    """이미지를 기반으로 유사한 가구 추천"""
    if not os.path.exists(generated_img_path):
        raise HTTPException(status_code=400, detail="이미지 파일이 존재하지 않습니다.")

    img = Image.open(generated_img_path).convert("RGB")

    # ✅ CNN 특징 벡터 추출 (512차원)
    with torch.no_grad():
        cnn_features = resnet(cnn_transform(img).unsqueeze(0).to(device)).squeeze().detach().cpu().numpy()[:512]

    # ✅ ViT 특징 벡터 추출 (512차원)
    vit_inputs = feature_extractor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        vit_features = vit_model(**vit_inputs).last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()[:512]

    # ✅ CLIP 특징 벡터 추출 (512차원)
    clip_img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        clip_features = clip_model.encode_image(clip_img_tensor).detach().cpu().numpy().flatten()

    # ✅ Texture 특징 벡터 (랜덤값 사용)
    texture_features = np.random.rand(512)

    # ✅ 벡터 결합
    query_vector = 0.3 * cnn_features + 0.3 * vit_features + 0.4 * clip_features + 0.2 * texture_features

    # ✅ FAISS 검색
    results = search_faiss(query_vector, top_k * 2)

    # ✅ 가격 필터링 (가격 변환 오류 방지)
    filtered_results = []
    for item in results:
        try:
            numeric_price = int(item["price"].replace("원", "").replace(",", ""))
            if min_price <= numeric_price <= max_price:
                filtered_results.append(item)
        except ValueError:
            print(f"⚠️ 가격 변환 실패: {item.get('price', 'Unknown')}")
    
    return {"recommendations": filtered_results[:top_k]}

@app.get("/recommend")
async def recommend_furniture_get(generated_img_path: str, min_price: int, max_price: int, top_k: int = 4):
    """GET 요청을 통한 가구 추천"""
    return get_recommendations(generated_img_path, min_price, max_price, top_k)

@app.post("/recommend")
async def recommend_furniture_post(request: RecommendationRequest = Body(...)):
    """POST 요청을 통한 가구 추천"""
    return get_recommendations(request.generated_img_path, request.min_price, request.max_price, request.top_k)

@app.get("/faiss_status")
async def check_faiss_status():
    """FAISS 인덱스 상태 확인"""
    if faiss_index is None:
        return {"status": "FAISS 인덱스가 로드되지 않았습니다."}
    elif faiss_index.ntotal == 0:
        return {"status": "FAISS 인덱스가 비어 있습니다."}
    return {"status": "FAISS 인덱스가 정상적으로 로드되었습니다.", "total_vectors": faiss_index.ntotal}

if __name__ == "__main__":
    load_faiss_index()
    uvicorn.run(app, host="0.0.0.0", port=8000)
