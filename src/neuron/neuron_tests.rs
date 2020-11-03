use crate::neuron::{ChargeCycle, Neuron, Neuronic, NeuronicInput, NeuronicSensor, SynapticType};
use std::rc::Rc;

/// Utility method that compares f32 to
/// three decimal places
fn cmp_f32(f1: f32, f2: f32) {
    assert_eq!(
        (f1 * 1000.).floor(),
        (f2 * 1000.).floor(),
        "{} does not equal {}",
        f1,
        f2
    );
}

#[test]
fn test_neuronic_sensor() {
    let sensor = NeuronicSensor::new();

    // Test this sensor to make sure it initialized correctly
    cmp_f32(sensor.get_measure(ChargeCycle::Even), 0.0);
    cmp_f32(sensor.get_measure(ChargeCycle::Odd), 0.0);

    let measure1 = 0.6;

    sensor.set_measure(measure1);

    // Test this sensor for both cycles
    cmp_f32(sensor.get_measure(ChargeCycle::Even), measure1);
    cmp_f32(sensor.get_measure(ChargeCycle::Odd), measure1);

    let measure2 = 0.4;

    sensor.set_measure(measure2);

    // Test this sensor after we've changed from previously set value
    cmp_f32(sensor.get_measure(ChargeCycle::Even), measure2);
    cmp_f32(sensor.get_measure(ChargeCycle::Odd), measure2);
}

#[test]
fn test_create_synapse() {
    let neuron = Neuron::new(10., 2., 2.);

    assert_eq!(neuron.synapses.borrow().len(), 0);

    neuron.create_synapse(2., SynapticType::Excitatory, Rc::new(NeuronicSensor::new()));

    assert_eq!(neuron.synapses.borrow().len(), 1);

    neuron.create_synapse(0., SynapticType::Inhibitory, Rc::new(NeuronicSensor::new()));

    assert_eq!(neuron.synapses.borrow().len(), 2);

    neuron.create_synapse(3., SynapticType::Excitatory, Rc::new(NeuronicSensor::new()));

    assert_eq!(neuron.synapses.borrow().len(), 3);

    neuron.create_synapse(0., SynapticType::Inhibitory, Rc::new(NeuronicSensor::new()));

    assert_eq!(neuron.synapses.borrow().len(), 4);
}

#[test]
fn test_run_static_cycle() {
    let fire_threshold = 10.;
    let max_weight = 8.;
    let learning_constant = 3.;

    let neuron = Neuron::new(fire_threshold, max_weight, learning_constant);

    cmp_f32(neuron.get_measure(ChargeCycle::Even), 0.0);
    cmp_f32(neuron.get_measure(ChargeCycle::Odd), 0.0);

    // Here "s" stands for sensor
    let s1 = Rc::new(NeuronicSensor::new());
    let s2 = Rc::new(NeuronicSensor::new());
    let s3 = Rc::new(NeuronicSensor::new());
    let s4 = Rc::new(NeuronicSensor::new());

    let s1_weight = 6.;
    let s1_type = SynapticType::Excitatory;
    neuron.create_synapse(s1_weight, s1_type, Rc::clone(&s1) as Rc<dyn NeuronicInput>);

    let s2_weight = 7.5;
    let s2_type = SynapticType::Excitatory;
    neuron.create_synapse(s2_weight, s2_type, Rc::clone(&s2) as Rc<dyn NeuronicInput>);

    let s3_weight = 3.;
    let s3_type = SynapticType::Inhibitory;
    neuron.create_synapse(s3_weight, s3_type, Rc::clone(&s3) as Rc<dyn NeuronicInput>);

    let s4_weight = 7.;
    let s4_type = SynapticType::Inhibitory;
    neuron.create_synapse(s4_weight, s4_type, Rc::clone(&s4) as Rc<dyn NeuronicInput>);

    // Test basic firing
    let s1_measure = 0.9;
    let s2_measure = 0.8;
    let s3_measure = 0.3;
    let s4_measure = 0.4;

    s1.set_measure(s1_measure);
    s2.set_measure(s2_measure);
    s3.set_measure(s3_measure);
    s4.set_measure(s4_measure);

    neuron.run_static_cycle(ChargeCycle::Even);

    cmp_f32(neuron.get_measure(ChargeCycle::Even), s2_measure);
    cmp_f32(neuron.get_measure(ChargeCycle::Odd), 0.0);

    // Test firing with inhibition on top that can be overcome
    let last_fire = s2_measure;

    let s1_measure = 0.8;
    let s2_measure = 0.7;
    let s3_measure = 0.9;
    let s4_measure = 0.4;

    s1.set_measure(s1_measure);
    s2.set_measure(s2_measure);
    s3.set_measure(s3_measure);
    s4.set_measure(s4_measure);

    neuron.run_static_cycle(ChargeCycle::Odd);

    cmp_f32(neuron.get_measure(ChargeCycle::Even), last_fire);
    cmp_f32(neuron.get_measure(ChargeCycle::Odd), s2_measure);

    // Test firing with a lower order inhibition that can be overcome
    let last_fire = s2_measure;

    let s1_measure = 0.7;
    let s2_measure = 0.9;
    let s3_measure = 0.8;
    let s4_measure = 0.4;

    s1.set_measure(s1_measure);
    s2.set_measure(s2_measure);
    s3.set_measure(s3_measure);
    s4.set_measure(s4_measure);

    neuron.run_static_cycle(ChargeCycle::Even);

    cmp_f32(neuron.get_measure(ChargeCycle::Even), s1_measure);
    cmp_f32(neuron.get_measure(ChargeCycle::Odd), last_fire);

    // Test firing with inhibition on top that can't be overcome
    let last_fire = s1_measure;

    let s1_measure = 0.7;
    let s2_measure = 0.9;
    let s3_measure = 0.4;
    let s4_measure = 0.95;

    s1.set_measure(s1_measure);
    s2.set_measure(s2_measure);
    s3.set_measure(s3_measure);
    s4.set_measure(s4_measure);

    neuron.run_static_cycle(ChargeCycle::Odd);

    cmp_f32(neuron.get_measure(ChargeCycle::Even), last_fire);
    cmp_f32(neuron.get_measure(ChargeCycle::Odd), 0.0);

    // Test firing with a lower inhibition that can't be overcome
    let s1_measure = 0.7;
    let s2_measure = 0.9;
    let s3_measure = 0.4;
    let s4_measure = 0.8;

    s1.set_measure(s1_measure);
    s2.set_measure(s2_measure);
    s3.set_measure(s3_measure);
    s4.set_measure(s4_measure);

    neuron.run_static_cycle(ChargeCycle::Even);

    cmp_f32(neuron.get_measure(ChargeCycle::Even), 0.0);
    cmp_f32(neuron.get_measure(ChargeCycle::Odd), 0.0);
}

/// This test basically has the same form as the previous test,
/// but now we're altering synapses
#[test]
fn test_update_synapses() {
    let fire_threshold = 10.;
    let max_weight = 8.;
    let learning_constant = 0.1;

    let neuron = Neuron::new(fire_threshold, max_weight, learning_constant);

    // Here "s" stands for sensor
    let s1 = Rc::new(NeuronicSensor::new());
    let s2 = Rc::new(NeuronicSensor::new());
    let s3 = Rc::new(NeuronicSensor::new());
    let s4 = Rc::new(NeuronicSensor::new());

    let mut s1_weight = 6.;
    let s1_type = SynapticType::Excitatory;
    neuron.create_synapse(s1_weight, s1_type, Rc::clone(&s1) as Rc<dyn NeuronicInput>);

    let mut s2_weight = 7.5;
    let s2_type = SynapticType::Excitatory;
    neuron.create_synapse(s2_weight, s2_type, Rc::clone(&s2) as Rc<dyn NeuronicInput>);

    let mut s3_weight = 3.;
    let s3_type = SynapticType::Inhibitory;
    neuron.create_synapse(s3_weight, s3_type, Rc::clone(&s3) as Rc<dyn NeuronicInput>);

    let mut s4_weight = 7.;
    let s4_type = SynapticType::Inhibitory;
    neuron.create_synapse(s4_weight, s4_type, Rc::clone(&s4) as Rc<dyn NeuronicInput>);

    let s1_measure = 0.8;
    let s2_measure = 0.9;
    let s3_measure = 0.9;
    let s4_measure = 0.4;

    s1.set_measure(s1_measure);
    s2.set_measure(s2_measure);
    s3.set_measure(s3_measure);
    s4.set_measure(s4_measure);

    neuron.run_static_cycle(ChargeCycle::Even);
    neuron.update_synapses(ChargeCycle::Even);

    let s1_calculated_weight =
        s1_weight + (learning_constant * (max_weight - s1_weight) * s1_measure);
    let s2_calculated_weight = s2_weight
        + (learning_constant * (max_weight - s2_weight) * ((2. * s1_measure) - s2_measure));
    let s3_calculated_weight = s3_weight
        + (learning_constant * (max_weight - s3_weight) * ((2. * s1_measure) - s3_measure));
    let s4_calculated_weight =
        s4_weight + (learning_constant * (max_weight - s4_weight) * s4_measure);

    {
        let synapses = neuron.synapses.borrow();
        s1_weight = synapses.get(0).unwrap().weight;
        s2_weight = synapses.get(1).unwrap().weight;
        s3_weight = synapses.get(2).unwrap().weight;
        s4_weight = synapses.get(3).unwrap().weight;
    };

    cmp_f32(s1_calculated_weight, s1_weight);
    cmp_f32(s2_calculated_weight, s2_weight);
    cmp_f32(s3_calculated_weight, s3_weight);
    cmp_f32(s4_calculated_weight, s4_weight);

    let s1_measure = 0.8;
    let s2_measure = 0.9;
    let s3_measure = 0.9;
    let s4_measure = 0.9;

    s1.set_measure(s1_measure);
    s2.set_measure(s2_measure);
    s3.set_measure(s3_measure);
    s4.set_measure(s4_measure);

    neuron.run_static_cycle(ChargeCycle::Odd);
    neuron.update_synapses(ChargeCycle::Odd);

    let s1_calculated_weight =
        s1_weight + (learning_constant * (max_weight - s1_weight) * -1. * s1_measure);
    let s2_calculated_weight =
        s2_weight + (learning_constant * (max_weight - s2_weight) * -1. * s2_measure);
    let s3_calculated_weight =
        s3_weight + (learning_constant * (max_weight - s3_weight) * -1. * s3_measure);
    let s4_calculated_weight =
        s4_weight + (learning_constant * (max_weight - s4_weight) * -1. * s4_measure);

    {
        let synapses = neuron.synapses.borrow();
        s1_weight = synapses.get(0).unwrap().weight;
        s2_weight = synapses.get(1).unwrap().weight;
        s3_weight = synapses.get(2).unwrap().weight;
        s4_weight = synapses.get(3).unwrap().weight;
    };

    cmp_f32(s1_calculated_weight, s1_weight);
    cmp_f32(s2_calculated_weight, s2_weight);
    cmp_f32(s3_calculated_weight, s3_weight);
    cmp_f32(s4_calculated_weight, s4_weight);

    let s1_measure = 0.9;
    let s2_measure = 0.9;
    let s3_measure = 0.4;
    let s4_measure = 0.4;

    s1.set_measure(s1_measure);
    s2.set_measure(s2_measure);
    s3.set_measure(s3_measure);
    s4.set_measure(s4_measure);

    neuron.run_static_cycle(ChargeCycle::Even);
    neuron.update_synapses(ChargeCycle::Even);

    let s1_calculated_weight =
        s1_weight + (learning_constant * (max_weight - s1_weight) * s1_measure);
    let s2_calculated_weight =
        s2_weight + (learning_constant * (max_weight - s2_weight) * s2_measure);
    let s3_calculated_weight =
        s3_weight + (learning_constant * (max_weight - s3_weight) * s3_measure);
    let s4_calculated_weight =
        s4_weight + (learning_constant * (max_weight - s4_weight) * s4_measure);

    {
        let synapses = neuron.synapses.borrow();
        s1_weight = synapses.get(0).unwrap().weight;
        s2_weight = synapses.get(1).unwrap().weight;
        s3_weight = synapses.get(2).unwrap().weight;
        s4_weight = synapses.get(3).unwrap().weight;
    };

    cmp_f32(s1_calculated_weight, s1_weight);
    cmp_f32(s2_calculated_weight, s2_weight);
    cmp_f32(s3_calculated_weight, s3_weight);
    cmp_f32(s4_calculated_weight, s4_weight);
}

#[test]
fn test_multiple_neurons() {
    let fire_threshold = 10.;
    let max_weight = 8.;
    let learning_constant = 3.;

    let n1 = Rc::new(Neuron::new(fire_threshold, max_weight, learning_constant));

    let n2 = Rc::new(Neuron::new(fire_threshold, max_weight, learning_constant));

    // Here "s" stands for sensor
    let s1 = Rc::new(NeuronicSensor::new());
    let s2 = Rc::new(NeuronicSensor::new());
    let s3 = Rc::new(NeuronicSensor::new());
    let s4 = Rc::new(NeuronicSensor::new());

    let s1_type = SynapticType::Excitatory;
    let s2_type = SynapticType::Excitatory;
    let s3_type = SynapticType::Excitatory;
    let s4_type = SynapticType::Inhibitory;

    let n1_s1_weight = 6.;
    let n1_s2_weight = 7.;
    let n1_s3_weight = 4.;
    let n1_s4_weight = 8.;
    n1.create_synapse(
        n1_s1_weight,
        s1_type,
        Rc::clone(&s1) as Rc<dyn NeuronicInput>,
    );
    n1.create_synapse(
        n1_s2_weight,
        s2_type,
        Rc::clone(&s2) as Rc<dyn NeuronicInput>,
    );
    n1.create_synapse(
        n1_s3_weight,
        s3_type,
        Rc::clone(&s3) as Rc<dyn NeuronicInput>,
    );
    n1.create_synapse(
        n1_s4_weight,
        s4_type,
        Rc::clone(&s4) as Rc<dyn NeuronicInput>,
    );

    let n2_s1_weight = 2.;
    let n2_s2_weight = 4.;
    let n2_s3_weight = 8.;
    let n2_s4_weight = 8.;
    n2.create_synapse(
        n2_s1_weight,
        s1_type,
        Rc::clone(&s1) as Rc<dyn NeuronicInput>,
    );
    n2.create_synapse(
        n2_s2_weight,
        s2_type,
        Rc::clone(&s2) as Rc<dyn NeuronicInput>,
    );
    n2.create_synapse(
        n2_s3_weight,
        s3_type,
        Rc::clone(&s3) as Rc<dyn NeuronicInput>,
    );
    n2.create_synapse(
        n2_s4_weight,
        s4_type,
        Rc::clone(&s4) as Rc<dyn NeuronicInput>,
    );

    let n3 = Neuron::new(fire_threshold, max_weight, learning_constant);

    let n3_n1_weight = 8.;
    let n3_n2_weight = 8.;
    n3.create_synapse(
        n3_n1_weight,
        SynapticType::Excitatory,
        Rc::clone(&n1) as Rc<dyn NeuronicInput>,
    );
    n3.create_synapse(
        n3_n2_weight,
        SynapticType::Excitatory,
        Rc::clone(&n2) as Rc<dyn NeuronicInput>,
    );

    let s1_measure = 0.9;
    let s2_measure = 0.8;
    let s3_measure = 0.7;
    let s4_measure = 0.2;
    s1.set_measure(s1_measure);
    s2.set_measure(s2_measure);
    s3.set_measure(s3_measure);
    s4.set_measure(s4_measure);

    let cycle = ChargeCycle::Even;
    n1.run_static_cycle(cycle);
    n2.run_static_cycle(cycle);
    n3.run_static_cycle(cycle);

    cmp_f32(n1.get_measure(cycle), s2_measure);
    cmp_f32(n1.get_measure(cycle.prev_cycle()), 0.0);
    cmp_f32(n2.get_measure(cycle), s3_measure);
    cmp_f32(n2.get_measure(cycle.prev_cycle()), 0.0);
    cmp_f32(n3.get_measure(cycle), 0.0);
    cmp_f32(n3.get_measure(cycle.prev_cycle()), 0.0);

    let n1_last_fired = s2_measure;
    let n2_last_fired = s3_measure;

    let s1_measure = 0.6;
    let s2_measure = 0.8;
    let s3_measure = 0.7;
    let s4_measure = 0.9;
    s1.set_measure(s1_measure);
    s2.set_measure(s2_measure);
    s3.set_measure(s3_measure);
    s4.set_measure(s4_measure);

    let cycle = cycle.next_cycle();
    n1.run_cycle(cycle);
    n2.run_cycle(cycle);
    n3.run_cycle(cycle);

    cmp_f32(n1.get_measure(cycle), 0.0);
    cmp_f32(n1.get_measure(cycle.prev_cycle()), n1_last_fired);
    cmp_f32(n2.get_measure(cycle), 0.0);
    cmp_f32(n2.get_measure(cycle.prev_cycle()), n2_last_fired);
    cmp_f32(n3.get_measure(cycle), n2_last_fired);
    cmp_f32(n3.get_measure(cycle.prev_cycle()), 0.0);
}

#[test]
fn test_clear() {
    let fire_threshold = 10.;
    let max_weight = 8.;
    let learning_constant = 3.;

    let neuron = Neuron::new(fire_threshold, max_weight, learning_constant);

    cmp_f32(neuron.get_measure(ChargeCycle::Even), 0.0);
    cmp_f32(neuron.get_measure(ChargeCycle::Odd), 0.0);

    // Here "s" stands for sensor
    let s1 = Rc::new(NeuronicSensor::new());
    let s2 = Rc::new(NeuronicSensor::new());
    let s3 = Rc::new(NeuronicSensor::new());
    let s4 = Rc::new(NeuronicSensor::new());

    let s1_weight = 6.;
    let s1_type = SynapticType::Excitatory;
    neuron.create_synapse(s1_weight, s1_type, Rc::clone(&s1) as Rc<dyn NeuronicInput>);

    let s2_weight = 7.5;
    let s2_type = SynapticType::Excitatory;
    neuron.create_synapse(s2_weight, s2_type, Rc::clone(&s2) as Rc<dyn NeuronicInput>);

    let s3_weight = 3.;
    let s3_type = SynapticType::Inhibitory;
    neuron.create_synapse(s3_weight, s3_type, Rc::clone(&s3) as Rc<dyn NeuronicInput>);

    let s4_weight = 7.;
    let s4_type = SynapticType::Inhibitory;
    neuron.create_synapse(s4_weight, s4_type, Rc::clone(&s4) as Rc<dyn NeuronicInput>);

    // Test basic firing
    let s1_measure = 0.9;
    let s2_measure = 0.8;
    let s3_measure = 0.3;
    let s4_measure = 0.4;

    s1.set_measure(s1_measure);
    s2.set_measure(s2_measure);
    s3.set_measure(s3_measure);
    s4.set_measure(s4_measure);

    neuron.run_static_cycle(ChargeCycle::Even);
    neuron.clear();

    cmp_f32(neuron.get_measure(ChargeCycle::Even), 0.0);
    cmp_f32(neuron.get_measure(ChargeCycle::Odd), 0.0);

    // Test firing with inhibition on top that can be overcome

    let s1_measure = 0.8;
    let s2_measure = 0.7;
    let s3_measure = 0.9;
    let s4_measure = 0.4;

    s1.set_measure(s1_measure);
    s2.set_measure(s2_measure);
    s3.set_measure(s3_measure);
    s4.set_measure(s4_measure);

    neuron.run_static_cycle(ChargeCycle::Odd);
    neuron.clear();

    cmp_f32(neuron.get_measure(ChargeCycle::Even), 0.0);
    cmp_f32(neuron.get_measure(ChargeCycle::Odd), 0.0);

    // Test firing with a lower order inhibition that can be overcome

    let s1_measure = 0.7;
    let s2_measure = 0.9;
    let s3_measure = 0.8;
    let s4_measure = 0.4;

    s1.set_measure(s1_measure);
    s2.set_measure(s2_measure);
    s3.set_measure(s3_measure);
    s4.set_measure(s4_measure);

    neuron.run_static_cycle(ChargeCycle::Even);
    neuron.clear();

    cmp_f32(neuron.get_measure(ChargeCycle::Even), 0.0);
    cmp_f32(neuron.get_measure(ChargeCycle::Odd), 0.0);

    // Test firing with inhibition on top that can't be overcome

    let s1_measure = 0.7;
    let s2_measure = 0.9;
    let s3_measure = 0.4;
    let s4_measure = 0.95;

    s1.set_measure(s1_measure);
    s2.set_measure(s2_measure);
    s3.set_measure(s3_measure);
    s4.set_measure(s4_measure);

    neuron.run_static_cycle(ChargeCycle::Odd);
    neuron.clear();

    cmp_f32(neuron.get_measure(ChargeCycle::Even), 0.0);
    cmp_f32(neuron.get_measure(ChargeCycle::Odd), 0.0);

    // Test firing with a lower inhibition that can't be overcome
    let s1_measure = 0.7;
    let s2_measure = 0.9;
    let s3_measure = 0.4;
    let s4_measure = 0.8;

    s1.set_measure(s1_measure);
    s2.set_measure(s2_measure);
    s3.set_measure(s3_measure);
    s4.set_measure(s4_measure);

    neuron.run_static_cycle(ChargeCycle::Even);
    neuron.clear();

    cmp_f32(neuron.get_measure(ChargeCycle::Even), 0.0);
    cmp_f32(neuron.get_measure(ChargeCycle::Odd), 0.0);
}
