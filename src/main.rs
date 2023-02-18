use tiny_http::{Response, Server};

/*
fn main() {
    let server = Server::http("0.0.0.0:8000").unwrap();

    println!("Server available on http://0.0.0.0:8000");
    for request in server.incoming_requests() {
        println!(
            "received request! method: {:?}, url: {:?}, headers: {:?}",
            request.method(),
            request.url(),
            request.headers()
        );

        let response = Response::from_string("hello world");
        let _ = request.respond(response);
    }
}
*/

#[pollster::main]
async fn main() {
    fourier::run().await;
}
