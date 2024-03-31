const PROTO_PATH: &str = "../protos/embeddings.proto";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos(PROTO_PATH)
        .unwrap_or_else(|e| panic!("Failed to compile protos {:?}", e));
    Ok(())
}
