use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::rc::Rc;

/// For better documentation of everything, see Eywa and Aesir
/// crates.  Most things are very similar between this library
/// and those libraries

#[derive(Copy, Clone)]
pub enum ChargeCycle {
    Even,
    Odd,
}

impl ChargeCycle {
    // Note that prev_cycle and next_cycle
    // do the exact same thing
    pub fn next_cycle(&self) -> ChargeCycle {
        self.prev_cycle()
    }

    pub fn prev_cycle(&self) -> ChargeCycle {
        match self {
            ChargeCycle::Even => ChargeCycle::Odd,
            ChargeCycle::Odd => ChargeCycle::Even,
        }
    }
}

/// All neurons implement this trait
pub trait Neuronic {
    /// Cycle where learning occurs, i.e. synaptic-weight updates
    fn run_cycle(&self, cycle: ChargeCycle) {
        self.run_static_cycle(cycle);
        self.update_synapses(cycle);
    }

    /// Cycle where learning does not occur, and simply processes IO
    fn run_static_cycle(&self, cycle: ChargeCycle);

    /// Update synapses based on current measure
    fn update_synapses(&self, cycle: ChargeCycle);

    /// Clears the measure of this Neuron, to clear out any
    /// residual inputs
    fn clear(&self);

    /// Creates a synapse with the neuronic input
    fn create_synapse(
        &self,
        starting_weight: f32,
        synaptic_type: SynapticType,
        input: Rc<dyn NeuronicInput>,
    );
}

/// Any object that functions as a pre-synaptic input
/// to a neuron must implement this trait
pub trait NeuronicInput {
    fn get_measure(&self, cycle: ChargeCycle) -> f32;
}

#[derive(Copy, Clone)]
pub struct Impulse {
    measure: f32,
    weight: f32,
}

impl Impulse {
    pub fn new(measure: f32, weight: f32) -> Impulse {
        Impulse { measure, weight }
    }
}

/// These traits must be implemented for Impulse to behave well
/// in impulse_heap
impl Eq for Impulse {}

impl PartialEq for Impulse {
    fn eq(&self, other: &Self) -> bool {
        self.measure.eq(&other.measure)
    }
}

impl PartialOrd for Impulse {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.measure.partial_cmp(&other.measure)
    }
}

impl Ord for Impulse {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.measure > other.measure {
            Ordering::Greater
        } else if self.measure < other.measure {
            Ordering::Less
        } else {
            Ordering::Equal
        }
    }
}

/// Basic synapse type and synpase
#[derive(Clone, Copy)]
pub enum SynapticType {
    Excitatory,
    Inhibitory,
}

pub struct Synapse {
    weight: f32,
    synaptic_type: SynapticType,
    pre_synaptic_neuron: Rc<dyn NeuronicInput>,
    last_impulse: Option<Impulse>,
}

impl Synapse {
    pub fn new(weight: f32, synaptic_type: SynapticType, neuron: Rc<dyn NeuronicInput>) -> Synapse {
        Synapse {
            weight,
            synaptic_type,
            pre_synaptic_neuron: neuron,
            last_impulse: None,
        }
    }

    pub fn generate_impulse(&mut self, cycle: ChargeCycle) -> Impulse {
        let measure = self.pre_synaptic_neuron.get_measure(cycle.prev_cycle());

        let impulse = match self.synaptic_type {
            SynapticType::Excitatory => Impulse::new(measure, self.weight),
            SynapticType::Inhibitory => Impulse::new(measure, -1.0 * self.weight),
        };

        self.last_impulse = Some(impulse);

        impulse
    }
}

/// Stores the Neuron's measure for different charge cycles
pub struct InternalMeasure(f32, f32);

impl InternalMeasure {
    pub fn new() -> InternalMeasure {
        InternalMeasure(0.0, 0.0)
    }

    pub fn set_measure(&mut self, cycle: ChargeCycle, measure: f32) {
        match cycle {
            ChargeCycle::Even => self.0 = measure,
            ChargeCycle::Odd => self.1 = measure,
        }
    }

    pub fn get_measure(&self, cycle: ChargeCycle) -> f32 {
        match cycle {
            ChargeCycle::Even => self.0,
            ChargeCycle::Odd => self.1,
        }
    }

    pub fn clear(&mut self) {
        self.0 = 0.0;
        self.1 = 0.0;
    }
}

/// In this library, due to the post-synaptic neuron owning synapses
/// There isn't a distinction between a plastic neuron and an actuator
/// neuron.  And a SensorNeuron is basically anything that only implements
/// NeuronicInput, so this simplifies implementation a butt-ton
pub struct Neuron {
    fire_threshold: f32,
    max_synapse_weight: f32,
    learning_constant: f32,
    synapses: RefCell<Vec<Synapse>>,
    internal_measure: RefCell<InternalMeasure>,
}

impl Neuron {
    pub fn new(fire_threshold: f32, max_synapse_weight: f32, learning_constant: f32) -> Neuron {
        Neuron {
            fire_threshold,
            max_synapse_weight,
            learning_constant,
            synapses: RefCell::new(Vec::new()),
            internal_measure: RefCell::new(InternalMeasure::new()),
        }
    }
}

impl NeuronicInput for Neuron {
    fn get_measure(&self, cycle: ChargeCycle) -> f32 {
        self.internal_measure.borrow().get_measure(cycle)
    }
}

impl Neuronic for Neuron {
    fn run_static_cycle(&self, cycle: ChargeCycle) {
        let mut synapses = self.synapses.borrow_mut();
        let mut internal_measure = self.internal_measure.borrow_mut();

        let mut impulse_heap: BinaryHeap<Impulse> = BinaryHeap::new();

        // Throw all impulses into the heap
        for synapse in synapses.iter_mut() {
            impulse_heap.push(synapse.generate_impulse(cycle));
        }

        // Get largest value impulses until the
        // aggregate weight surpasses the fire_threshold
        let mut total_weight = 0.0;

        loop {
            if let Some(impulse) = impulse_heap.pop() {
                total_weight += impulse.weight;

                if total_weight >= self.fire_threshold {
                    internal_measure.set_measure(cycle, impulse.measure);
                    break;
                }
            } else {
                internal_measure.set_measure(cycle, 0.0);
                break;
            }
        }
    }

    /// This is about the most basic update mechanism possible.
    /// Basically just a spring weighted by a measure
    fn update_synapses(&self, cycle: ChargeCycle) {
        let fired_measure = self.internal_measure.borrow().get_measure(cycle);

        for synapse in self.synapses.borrow_mut().iter_mut() {
            let synapse_measure = synapse.last_impulse.unwrap().measure;

            if synapse_measure < fired_measure {
                synapse.weight += self.learning_constant
                    * (self.max_synapse_weight - synapse.weight)
                    * synapse_measure;
            } else {
                synapse.weight += self.learning_constant
                    * (self.max_synapse_weight - synapse.weight)
                    * ((2.0 * fired_measure) - synapse_measure);
            }

            if synapse.weight > self.max_synapse_weight {
                synapse.weight = self.max_synapse_weight - 0.1; // Just under max weight so that it still updates
            } else if synapse.weight < 0.0 {
                synapse.weight = 0.0;
            }
        }
    }

    fn clear(&self) {
        self.internal_measure.borrow_mut().clear();
    }

    fn create_synapse(
        &self,
        starting_weight: f32,
        synaptic_type: SynapticType,
        input: Rc<dyn NeuronicInput>,
    ) {
        self.synapses
            .borrow_mut()
            .push(Synapse::new(starting_weight, synaptic_type, input));
    }
}

/// A simple sensor that can be set
/// and implements NeuronicInput
pub struct NeuronicSensor {
    measure: RefCell<f32>
}

impl NeuronicSensor {
    pub fn new() -> NeuronicSensor {
        NeuronicSensor {
            measure: RefCell::new(0.0)
        }
    }

    pub fn set_measure(&self, measure: f32) {
        *self.measure.borrow_mut() = measure;
    }
}

impl NeuronicInput for NeuronicSensor {
    fn get_measure(&self, _cycle: ChargeCycle) -> f32 {
        *self.measure.borrow()
    }
}

#[cfg(test)]
mod neuron_tests;
