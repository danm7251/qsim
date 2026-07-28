#[cfg(feature = "trace")]
#[path ="../src/trace.rs"]
mod trace;

use qsim::{gates::Gate, state::State};
use rand::random;

/// An implementation of Deutsch's algorithm.
/// On each run the algorithm is given one of four functions at random.
/// In just one check it determines whether the function is balanced or constant.
/// In classical computing this would require two checks.

fn main() {
    #[cfg(feature = "trace")]
    let _guard = trace::init_tracing();

    // Generate a random integer between 1 and 4
    let n = (random::<u8>() % 4) + 1;
    // Use it to pick a Deutsch function
    let f = deutsch_function(n);
    // Run it through the Deutsch algorithm
    let outcome = deutsch_algorithm(&f);
    
    if outcome {
        assert!(n==2 || n==3);
        println!("f{} is balanced!", n);
    } else {
        assert!(n==1 || n==4);
        println!("f{} is constant!", n);
    }
}

fn deutsch_algorithm(f: &Vec<Gate>) -> bool {
    // Setup initial state
    let mut state = State::zero(2).unwrap();
    state.apply_gate(Gate::X { target: 1 }).expect("Failed to apply Gate::X to target=1");
    state.apply_gate(Gate::H { target: 0 }).expect("Failed to apply Gate::H to target=0");
    state.apply_gate(Gate::H { target: 1 }).expect("Failed to apply Gate::H to target=1");

    // Apply Deutsch function
    for &g in f {
        state.apply_gate(g).expect("Failed to apply Deutsch function");
    }

    // Transform and measure result
    state.apply_gate(Gate::H { target: 0 }).expect("Failed to apply Gate::H to target=0");
    state.measure(0)
}

fn deutsch_function(n: u8) -> Vec<Gate> {
    match n {
        // f1(a) = 0
        1 => vec![],
        // f2(a) = a
        2 => vec![Gate::CNOT { control: 0, target: 1 }],
        // f3(a) = !a
        3 => vec![Gate::CNOT { control: 0, target: 1 }, Gate::X { target: 1 }],
        // f4(a) = 1
        4 => vec![Gate::X { target: 1 }],
        _ => panic!("Invalid option, please pick f1(), f2(), f3() or f4()")
    }
}