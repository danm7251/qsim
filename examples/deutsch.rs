#[cfg(feature = "trace")]
#[path ="../src/trace.rs"]
mod trace;

use qsim::{api::Instruction, state::State};
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

fn deutsch_algorithm(f: &Vec<Instruction>) -> bool {
    // Setup initial state
    let mut state = State::zero(2).unwrap();
    state.execute(Instruction::X { q: 1 }).expect("Failed to apply Instruction::X to target=1");
    state.execute(Instruction::H { q: 0 }).expect("Failed to apply Instruction::H to target=0");
    state.execute(Instruction::H { q: 1 }).expect("Failed to apply Instruction::H to target=1");

    // Apply Deutsch function
    for &g in f {
        state.execute(g).expect("Failed to apply Deutsch function");
    }

    // Transform and measure result
    state.execute(Instruction::H { q: 0 }).expect("Failed to apply Instruction::H to target=0");
    state.measure(0).unwrap()
}

fn deutsch_function(n: u8) -> Vec<Instruction> {
    match n {
        // f1(a) = 0
        1 => vec![],
        // f2(a) = a
        2 => vec![Instruction::CNOT { q_c: 0, q_t: 1 }],
        // f3(a) = !a
        3 => vec![Instruction::CNOT { q_c: 0, q_t: 1 }, Instruction::X { q: 1 }],
        // f4(a) = 1
        4 => vec![Instruction::X { q: 1 }],
        _ => panic!("Invalid option, please pick f1(), f2(), f3() or f4()")
    }
}