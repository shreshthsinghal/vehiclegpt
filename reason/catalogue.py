"""Plain-language catalogue of the recorded vehicle signals.

Lets VehicleGPT answer "what is my X right now?" / "how has X changed?" for any recorded signal, not only the dozen
the playbooks were written for. Each entry: key -> (VSS path, friendly name, unit, words that mean it, value labels).
Only signals that the car actually records are listed; anything else is answered as "not recorded".
"""
from __future__ import annotations

import re

V = "Vehicle."
GEAR = {0: "Park", 1: "Reverse", 2: "Neutral", 3: "Drive"}
POWER = {0: "off", 1: "accessory mode", 2: "on (ready to drive)", 3: "charging"}
CAMERA = {0: "OK", 1: "degraded", 2: "blocked", 3: "offline"}
WIPER = {0: "off", 1: "intermittent", 2: "low", 3: "high"}
ROAD = {0: "city road", 1: "highway"}
ONOFF = {0: "off", 1: "on"}
YESNO = {0: "no", 1: "yes"}

CATALOGUE: dict[str, tuple] = {
    "speed": (V + "Speed", "speed", "km/h", r"\bspeed\b|how fast|km/?h", None),
    "odometer": (V + "TraveledDistance", "odometer", "km", r"odometer|odo\b|total (distance|km|kilomet)|mileage reading", None),
    "gear": (V + "Powertrain.Transmission.SelectedGear", "gear", "", r"\bgear\b|drive mode selector|in park|in drive|in reverse", GEAR),
    "power_state": (V + "Private.Vehicle.PowerState", "power state", "", r"power state|switched on|turned on|car (on|off)\b|ignition|ready to drive", POWER),
    "accel_pedal": (V + "Chassis.Accelerator.PedalPosition", "accelerator pedal", "%", r"accelerator|throttle|gas pedal", None),
    "brake_pedal": (V + "Chassis.Brake.PedalPosition", "brake pedal", "%", r"brake pedal|braking now|pressing the brake", None),
    "steering": (V + "Chassis.SteeringWheel.Angle", "steering angle", "°", r"steering", None),
    "soc": (V + "Powertrain.TractionBattery.StateOfCharge.Current", "battery charge", "%", r"\bsoc\b|state of charge|charge level|battery (level|percent|%|charge)|how much (battery|charge)", None),
    "range": (V + "Powertrain.TractionBattery.Range", "estimated range", "km", r"\brange\b|how far can|km left|distance left", None),
    "soh": (V + "Powertrain.TractionBattery.StateOfHealth", "battery health", "%", r"battery health|\bsoh\b|state of health|degradation", None),
    "batt_temp": (V + "Powertrain.TractionBattery.Temperature.Max", "battery temperature (hottest cell)", "°C", r"batter\w* temp|battery (heat|hot)|how hot is the battery", None),
    "batt_voltage": (V + "Powertrain.TractionBattery.CurrentVoltage", "high-voltage battery voltage", "V", r"(battery|pack|hv) voltage|high.voltage", None),
    "batt_current": (V + "Powertrain.TractionBattery.CurrentCurrent", "battery current", "A", r"battery current|\bamps?\b|current draw", None),
    "batt_power": (V + "Powertrain.TractionBattery.CurrentPower", "battery power", "W", r"battery power|power (draw|usage|use) now|how much power", None),
    "charging": (V + "Powertrain.TractionBattery.Charging.IsCharging", "charging", "", r"(is|am i|are we) (it |the car )?charging|charging (now|right now|status)|plugged in", YESNO),
    "power_limit": (V + "Private.TractionBattery.PowerLimit.Discharge", "available drive power", "kW", r"power limit|derat|limp|reduced power|available power", None),
    "cell_min": (V + "Private.TractionBattery.CellVoltage.Min", "lowest cell voltage", "V", r"cell voltage|weakest cell|cell imbalance", None),
    "motor_temp": (V + "Powertrain.ElectricMotor.Temperature", "motor temperature", "°C", r"motor (temp|heat|hot)|how hot is the motor|motor running hot", None),
    "motor_speed": (V + "Powertrain.ElectricMotor.Speed", "motor speed", "rpm", r"motor speed|\brpm\b|motor rpm", None),
    "torque": (V + "Private.ElectricMotor.Torque", "motor torque", "Nm", r"torque", None),
    "regen": (V + "Private.Powertrain.RegenPower", "regenerative braking power", "kW", r"\bregen", None),
    "coolant_temp": (V + "Private.Thermal.CoolantTemperature.Battery", "battery coolant temperature", "°C", r"coolant temp", None),
    "coolant_pump": (V + "Private.Thermal.CoolantPump.Speed", "coolant pump speed", "rpm", r"coolant pump|cooling pump|\bpump\b", None),
    "chiller": (V + "Private.Thermal.ChillerActive", "battery chiller", "", r"chiller|battery cooling (on|active)", ONOFF),
    "ac_on": (V + "Cabin.HVAC.IsAirConditioningActive", "air-conditioning", "", r"\b(is|was) (the )?(ac|a/c|air.?con\w*) on|\bac (on|status)\b|air.?con\w* (on|status)", ONOFF),
    "cabin_temp": (V + "Cabin.HVAC.AmbientAirTemperature", "cabin temperature", "°C", r"cabin temp|inside temp|temperature inside|in the car temp", None),
    "set_temp": (V + "Cabin.HVAC.Station.Row1.Driver.Temperature", "climate set temperature", "°C", r"set (temp|point)|climate set|thermostat|ac (set|temperature)", None),
    "fan": (V + "Cabin.HVAC.Station.Row1.Driver.FanSpeed", "fan speed", "%", r"\bfan\b|blower", None),
    "ac_power": (V + "Private.HVAC.CompressorPower", "AC compressor power", "kW", r"compressor|ac power", None),
    "outside_temp": (V + "Exterior.AirTemperature", "outside temperature", "°C", r"outside temp|ambient|outdoor|temperature outside|weather (now|today)|how hot is it", None),
    "wiper": (V + "Body.Windshield.Front.Wiping.Mode", "wipers", "", r"wiper|raining now|is it raining", WIPER),
    "low_beam": (V + "Body.Lights.Beam.Low.IsOn", "headlights (low beam)", "", r"headlight|head lamp|low beam|lights on", ONOFF),
    "road_type": (V + "Private.Road.Type", "road type", "", r"road type|(city|highway) (road )?now|what road", ROAD),
    "tyre_fl": (V + "Chassis.Axle.Row1.Wheel.Left.Tire.Pressure", "front-left tyre pressure", "kPa", r"front.left (tyre|tire)", None),
    "tyre_fr": (V + "Chassis.Axle.Row1.Wheel.Right.Tire.Pressure", "front-right tyre pressure", "kPa", r"front.right (tyre|tire)", None),
    "tyre_rl": (V + "Chassis.Axle.Row2.Wheel.Left.Tire.Pressure", "rear-left tyre pressure", "kPa", r"rear.left (tyre|tire)|back.left (tyre|tire)", None),
    "tyre_rr": (V + "Chassis.Axle.Row2.Wheel.Right.Tire.Pressure", "rear-right tyre pressure", "kPa", r"rear.right (tyre|tire)|back.right (tyre|tire)", None),
    "cruise": (V + "ADAS.CruiseControl.IsActive", "cruise control", "", r"cruise", ONOFF),
    "lka": (V + "Private.ADAS.LaneKeepAssist.IsActive", "lane-keep assist", "", r"lane.?keep|lane assist|\blka\b", ONOFF),
    "ldw": (V + "ADAS.LaneDepartureDetection.IsEnabled", "lane-departure warning", "", r"lane.?departure", ONOFF),
    "lane_conf": (V + "Private.ADAS.LaneMarkingConfidence", "lane-marking confidence", "%", r"lane.?marking|lane confidence", None),
    "abs": (V + "ADAS.ABS.IsEngaged", "ABS (anti-lock brakes)", "", r"\babs\b|anti.?lock", YESNO),
    "esc": (V + "ADAS.ESC.IsEngaged", "ESC (stability control)", "", r"\besc\b|stability control|traction control|skid", YESNO),
    "eba": (V + "ADAS.EBA.IsEngaged", "emergency brake assist", "", r"\beba\b|brake assist", YESNO),
    "camera": (V + "Private.Camera.Front.Status", "front camera", "", r"camera", CAMERA),
    "visibility": (V + "Private.Camera.Front.Visibility", "camera visibility", "%", r"visibility", None),
    "lv_voltage": (V + "LowVoltageBattery.CurrentVoltage", "12 V battery voltage", "V", r"12 ?v|12.volt|small battery|auxiliary battery|low.voltage", None),
}
TYRES = ["tyre_fl", "tyre_fr", "tyre_rl", "tyre_rr"]
SNAPSHOT = ["power_state", "gear", "speed", "soc", "range", "batt_temp", "motor_temp", "outside_temp", "cabin_temp",
            "ac_on", "charging", "lv_voltage", "lka", "camera", "wiper", "odometer"]


def find(question: str) -> list[str]:
    """Signal keys mentioned in a question (most specific first)."""
    q = question.lower()
    hits = [k for k, (_, _, _, pat, _) in CATALOGUE.items() if re.search(pat, q)]
    if re.search(r"\b(tyres?|tires?)\b", q) and not any(k in hits for k in TYRES):
        hits += TYRES
    for specific, generic in (("lv_voltage", "batt_voltage"), ("motor_speed", "speed"), ("ac_power", "ac_on"), ("cabin_temp", "batt_temp")):
        if specific in hits and generic in hits:
            hits.remove(generic)
    if "range" in hits and "soc" in hits:
        hits.remove("range"); hits.insert(1, "range")
    return list(dict.fromkeys(hits))


def label(key: str, value) -> str:
    """Human-readable value with unit or enum label."""
    _, _, unit, _, labels = CATALOGUE[key]
    if labels is not None:
        return labels.get(int(round(float(value))), str(value))
    v = round(float(value), 1)
    v = int(v) if float(v).is_integer() else v
    return f"{v} {unit}".strip() if unit not in ("%", "°", "°C") else f"{v}{unit if unit != '°C' else ' °C'}"
