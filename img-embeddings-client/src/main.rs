pub mod embeddings_proto {
    tonic::include_proto!("embeddings_proto");
}

use crate::embeddings_proto::embeddings_client::EmbeddingsClient;
use candle_core::{DType, Device, Tensor};
use hs_grpc_shared::tensor_proto;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client: EmbeddingsClient<tonic::transport::Channel> =
        EmbeddingsClient::connect("http://[::1]:50051").await?;
    println!("Starting client");
    let input_img_batch =
        candle_core::Tensor::ones((1, 3, 224, 224), DType::F32, &Device::Cpu).unwrap();
    let req: tensor_proto::TensorProto = input_img_batch.try_into().unwrap();

    let response = client.get_embeddings(req).await?;

    println!("RESPONSE={:?}", response);

    let o3: Tensor = response.into_inner().try_into().unwrap();

    Ok(())
}
