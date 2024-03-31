use candle_core::{DType, Device, Tensor};
use std::convert::TryInto;
use tensor_proto::ByteOrder;
use tensor_proto::Dtype;
use tensor_proto::TensorProto;

pub mod tensor_proto {
    tonic::include_proto!("tensor_proto");
}

impl TryInto<Tensor> for TensorProto {
    type Error = ();

    fn try_into(self) -> Result<Tensor, Self::Error> {
        let t: Vec<f32> = match self.dtype() {
            Dtype::F32 => self
                .content
                .iter()
                .map(|i| f32::from_le_bytes(<[u8; 4]>::try_from(i.as_slice()).unwrap()))
                .collect(),
            // _ => unimplemented!()
        };

        Tensor::from_vec(
            t,
            self.shape
                .iter()
                .map(|dx| *dx as usize)
                .collect::<Vec<usize>>(),
            &Device::Cpu,
        )
        .map_err(|_| panic!())
    }
}

impl TryInto<TensorProto> for Tensor {
    type Error = ();

    fn try_into(self) -> Result<TensorProto, Self::Error> {
        let dims = self.dims();
        let (content, dtype) = match self.dtype() {
            DType::F32 => (
                self.flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap()
                    .iter()
                    .map(|i: &f32| i.to_le_bytes().to_vec())
                    .collect(),
                Dtype::F32.into(),
            ),
            _ => unimplemented!(),
        };

        Ok(TensorProto {
            content,
            shape: dims.iter().map(|dx| *dx as u32).collect(),
            dtype,
            byte_order: ByteOrder::Le.into(),
        })
    }
}
