use crate::embeddings_proto::embeddings_server::{Embeddings, EmbeddingsServer};
use candle_core::Module;
use candle_core::Tensor;
use hs_grpc_shared::tensor_proto;
use hs_mm_embeddings::clip_vit_b_32_image;
use std::net::SocketAddr;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

pub mod embeddings_proto {
    tonic::include_proto!("embeddings_proto");
}
pub struct EmbeddingsService {
    shared_state: std::sync::Arc<std::sync::Mutex<clip_vit_b_32_image::model::VisionTransformer>>,
}

#[tonic::async_trait]
impl Embeddings for EmbeddingsService {
    async fn get_embeddings(
        &self,
        request: Request<tensor_proto::TensorProto>,
    ) -> Result<Response<tensor_proto::TensorProto>, Status> {
        if let Ok(image_model) = self.shared_state.lock() {
            println!("Got a request: {:?}", request);
            let image: Tensor = request.into_inner().try_into().unwrap();

            let image: Tensor = image_model.forward(&image).unwrap();

            let reply: tensor_proto::TensorProto = image.try_into().unwrap();
            Ok(Response::new(reply))
        } else {
            todo!()
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = "[::1]:50051".parse()?;
    println!("Starting Server");
    let image_model = clip_vit_b_32_image::model::build_model().unwrap();

    let shared_state = std::sync::Arc::new(std::sync::Mutex::new(image_model));

    // let service:EmbeddingsService = EmbeddingsService::default();
    let service: EmbeddingsService = EmbeddingsService { shared_state };
    Server::builder()
        .add_service(EmbeddingsServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
