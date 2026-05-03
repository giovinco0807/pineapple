use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub struct InferenceRequest {
    pub model: String,
    pub board: String,
    pub hand: String,
    pub opp_top: String,
    pub opp_mid: String,
    pub opp_bot: String,
    pub dead_cards: String,
}

#[derive(Deserialize, Debug)]
pub struct InferenceResponse {
    pub probs: Vec<Vec<f64>>,
    pub ev: f64,
    pub error: Option<String>,
}

pub struct InferenceClient {
    stream: TcpStream,
    reader: BufReader<TcpStream>,
}

impl InferenceClient {
    pub fn new(address: &str) -> std::io::Result<Self> {
        let stream = TcpStream::connect(address)?;
        let reader = BufReader::new(stream.try_clone()?);
        Ok(Self { stream, reader })
    }

    pub fn predict(&mut self, req: &InferenceRequest) -> std::io::Result<InferenceResponse> {
        let req_json = serde_json::to_string(req)?;
        self.stream.write_all(req_json.as_bytes())?;
        self.stream.write_all(b"\n")?;
        self.stream.flush()?;

        let mut response_str = String::new();
        self.reader.read_line(&mut response_str)?;
        
        let resp: InferenceResponse = serde_json::from_str(&response_str)?;
        Ok(resp)
    }
}
