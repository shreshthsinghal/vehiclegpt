"""Build the Nexus EV CAN database (DBC) in code and write it to twin/nexus_ev.dbc.

A DBC file is the "decoder sheet" for a CAN bus: for every message ID it says which bytes hold which
signal, and how to scale the raw number into a real value (e.g. raw 6150 x 0.01 = 61.50 km/h).
Building it with cantools (instead of typing it by hand) guarantees the file is valid.

Run:  python -m twin.dbc
"""
from __future__ import annotations

from pathlib import Path

import cantools
from cantools.database import Database, Message, Signal
from cantools.database.conversion import BaseConversion

DBC_PATH = Path(__file__).with_name("nexus_ev.dbc")

# VSS path each CAN signal maps to (used by the SENSE normaliser in Phase 2).
VSS_MAP: dict[str, str] = {}


def _sig(name, start, length, scale=1.0, offset=0.0, signed=False, unit="", minimum=None, maximum=None,
         vss=None, choices=None):
    conv = BaseConversion.factory(scale=scale, offset=offset, choices=choices, is_float=False)
    s = Signal(name=name, start=start, length=length, byte_order="little_endian", is_signed=signed,
               conversion=conv, minimum=minimum, maximum=maximum, unit=unit)
    if vss:
        VSS_MAP[name] = vss
    return s


GEAR = {0: "P", 1: "R", 2: "N", 3: "D"}
POWER_STATE = {0: "OFF", 1: "ACC", 2: "ON", 3: "CHARGING"}
WIPER = {0: "OFF", 1: "INTERVAL", 2: "LOW", 3: "HIGH"}
WARNING = {0: "NONE", 1: "BATTERY_TEMP_HIGH", 2: "REDUCED_POWER", 3: "CHECK_TYRE_PRESSURE",
           4: "LOW_12V_BATTERY", 5: "CAMERA_UNAVAILABLE"}


def build() -> Database:
    VSS_MAP.clear()
    P = "Vehicle.Private."
    msgs = [
        Message(0x0A0, "VCU_Motion", 8, [
            _sig("VehicleSpeed", 0, 16, 0.01, unit="km/h", minimum=0, maximum=250, vss="Vehicle.Speed"),
            _sig("AccelLong", 16, 16, 0.001, signed=True, unit="m/s^2", vss="Vehicle.Acceleration.Longitudinal"),
            _sig("AccelPedal", 32, 8, 0.4, unit="%", vss="Vehicle.Chassis.Accelerator.PedalPosition"),
            _sig("BrakePedal", 40, 8, 0.4, unit="%", vss="Vehicle.Chassis.Brake.PedalPosition"),
            _sig("Gear", 48, 4, vss="Vehicle.Powertrain.Transmission.SelectedGear", choices=GEAR),
            _sig("PowerState", 52, 4, vss=P + "Vehicle.PowerState", choices=POWER_STATE),
        ], senders=["VCU"]),
        Message(0x0A1, "VCU_Chassis", 8, [
            _sig("SteeringAngle", 0, 16, 0.1, signed=True, unit="deg", vss="Vehicle.Chassis.SteeringWheel.Angle"),
            _sig("AccelLat", 16, 16, 0.001, signed=True, unit="m/s^2", vss="Vehicle.Acceleration.Lateral"),
            _sig("Odometer", 32, 32, 0.1, unit="km", vss="Vehicle.TraveledDistance"),
        ], senders=["VCU"]),
        Message(0x110, "MCU_Status", 8, [
            _sig("MotorSpeed", 0, 16, 1, signed=True, unit="rpm", vss="Vehicle.Powertrain.ElectricMotor.Speed"),
            _sig("MotorTemp", 16, 8, 1, -40, unit="degC", vss="Vehicle.Powertrain.ElectricMotor.Temperature"),
            _sig("MotorTorque", 24, 16, 0.5, signed=True, unit="Nm", vss=P + "ElectricMotor.Torque"),
            _sig("RegenPower", 40, 16, 0.01, unit="kW", vss=P + "Powertrain.RegenPower"),
        ], senders=["MCU"]),
        Message(0x200, "BMS_Pack", 8, [
            _sig("PackVoltage", 0, 16, 0.1, unit="V", vss="Vehicle.Powertrain.TractionBattery.CurrentVoltage"),
            _sig("PackCurrent", 16, 16, 0.1, signed=True, unit="A", vss="Vehicle.Powertrain.TractionBattery.CurrentCurrent"),
            _sig("PackPower", 32, 16, 10, signed=True, unit="W", vss="Vehicle.Powertrain.TractionBattery.CurrentPower"),
            _sig("SoC", 48, 10, 0.1, unit="%", vss="Vehicle.Powertrain.TractionBattery.StateOfCharge.Current"),
        ], senders=["BMS"]),
        Message(0x201, "BMS_Thermal", 8, [
            _sig("TempAvg", 0, 8, 0.5, -40, unit="degC", vss="Vehicle.Powertrain.TractionBattery.Temperature.Average"),
            _sig("TempMax", 8, 8, 0.5, -40, unit="degC", vss="Vehicle.Powertrain.TractionBattery.Temperature.Max"),
            _sig("TempMin", 16, 8, 0.5, -40, unit="degC", vss="Vehicle.Powertrain.TractionBattery.Temperature.Min"),
            _sig("SoH", 24, 8, 0.5, unit="%", vss="Vehicle.Powertrain.TractionBattery.StateOfHealth"),
            _sig("SoCDisplayed", 32, 8, 0.5, unit="%", vss="Vehicle.Powertrain.TractionBattery.StateOfCharge.Displayed"),
            _sig("DischargeLimit", 40, 8, 1, unit="kW", vss=P + "TractionBattery.PowerLimit.Discharge"),
            _sig("ChargeLimit", 48, 8, 1, unit="kW"),
        ], senders=["BMS"]),
        Message(0x202, "BMS_Charge", 8, [
            _sig("Range", 0, 16, 0.1, unit="km", vss="Vehicle.Powertrain.TractionBattery.Range"),
            _sig("IsCharging", 16, 1, vss="Vehicle.Powertrain.TractionBattery.Charging.IsCharging"),
            _sig("ChargeCurrentDC", 17, 15, 0.1, unit="A", vss="Vehicle.Powertrain.TractionBattery.Charging.ChargeCurrent.DC"),
            _sig("ChargeCurrentAC", 32, 16, 0.1, unit="A", vss="Vehicle.Powertrain.TractionBattery.Charging.ChargeCurrent.AC"),
        ], senders=["BMS"]),
        Message(0x203, "BMS_CellStats", 8, [
            _sig("CellVmin", 0, 16, 0.001, unit="V", vss=P + "TractionBattery.CellVoltage.Min"),
            _sig("CellVmax", 16, 16, 0.001, unit="V", vss=P + "TractionBattery.CellVoltage.Max"),
        ], senders=["BMS"]),
        Message(0x300, "HVAC_Status", 8, [
            _sig("CompressorPower", 0, 16, 0.001, unit="kW", vss=P + "HVAC.CompressorPower"),
            _sig("HeaterPower", 16, 16, 0.001, unit="kW", vss=P + "HVAC.HeaterPower"),
            _sig("ACActive", 32, 1, vss="Vehicle.Cabin.HVAC.IsAirConditioningActive"),
            _sig("CabinTemp", 40, 8, 0.5, -40, unit="degC", vss="Vehicle.Cabin.HVAC.AmbientAirTemperature"),
            _sig("SetPoint", 48, 8, 0.5, unit="degC", vss="Vehicle.Cabin.HVAC.Station.Row1.Driver.Temperature"),
            _sig("FanSpeed", 56, 8, 1, unit="%", vss="Vehicle.Cabin.HVAC.Station.Row1.Driver.FanSpeed"),
        ], senders=["HVAC"]),
        Message(0x310, "BCM_Environment", 8, [
            _sig("ExteriorTemp", 0, 8, 0.5, -40, unit="degC", vss="Vehicle.Exterior.AirTemperature"),
            _sig("WiperMode", 8, 2, vss="Vehicle.Body.Windshield.Front.Wiping.Mode", choices=WIPER),
            _sig("LowBeam", 10, 1, vss="Vehicle.Body.Lights.Beam.Low.IsOn"),
            _sig("LVBatteryVoltage", 16, 16, 0.001, unit="V", vss="Vehicle.LowVoltageBattery.CurrentVoltage"),
        ], senders=["BCM"]),
        Message(0x320, "TPMS_Pressures", 8, [
            _sig("TyrePressFL", 0, 16, 0.1, unit="kPa", vss="Vehicle.Chassis.Axle.Row1.Wheel.Left.Tire.Pressure"),
            _sig("TyrePressFR", 16, 16, 0.1, unit="kPa", vss="Vehicle.Chassis.Axle.Row1.Wheel.Right.Tire.Pressure"),
            _sig("TyrePressRL", 32, 16, 0.1, unit="kPa", vss="Vehicle.Chassis.Axle.Row2.Wheel.Left.Tire.Pressure"),
            _sig("TyrePressRR", 48, 16, 0.1, unit="kPa", vss="Vehicle.Chassis.Axle.Row2.Wheel.Right.Tire.Pressure"),
        ], senders=["TPMS"]),
        Message(0x330, "ESC_Status", 8, [
            _sig("ABSActive", 0, 1, vss="Vehicle.ADAS.ABS.IsEngaged"),
            _sig("ESCActive", 1, 1, vss="Vehicle.ADAS.ESC.IsEngaged"),
            _sig("EBAActive", 2, 1, vss="Vehicle.ADAS.EBA.IsEngaged"),
        ], senders=["ESC"]),
        Message(0x3F0, "Cluster_Warning", 8, [
            _sig("WarningCode", 0, 8, choices=WARNING),
            _sig("WarningActive", 8, 1),
        ], senders=["IPC"]),
    ]
    # CAN-FD messages (64-byte payload): per-cell voltages and per-module temperatures.
    for block in range(6):  # 6 x 16 cells = 96 cells
        sigs = [_sig(f"Cell{block * 16 + i + 1:02d}V", i * 16, 16, 0.001, unit="V",
                     vss=f"{P}TractionBattery.Cell{block * 16 + i + 1}.Voltage") for i in range(16)]
        msgs.append(Message(0x2A0 + block, f"BMS_Cells_{block}", 32, sigs, is_fd=True, senders=["BMS"]))
    sigs = [_sig(f"Module{i + 1:02d}T", i * 8, 8, 0.5, -40, unit="degC",
                 vss=f"{P}TractionBattery.Module{i + 1}.Temperature") for i in range(32)]
    msgs.append(Message(0x2B0, "BMS_ModuleTemps", 32, sigs, is_fd=True, senders=["BMS"]))
    return Database(messages=msgs, version="1.0")


def write(path: Path = DBC_PATH) -> Path:
    db = build()
    path.write_text(db.as_dbc_string(), encoding="utf-8")
    return path


def load(path: Path = DBC_PATH) -> Database:
    if not path.exists():
        write(path)
    return cantools.database.load_file(str(path))


if __name__ == "__main__":
    p = write()
    db = load(p)
    n = sum(len(m.signals) for m in db.messages)
    print(f"wrote {p} : {len(db.messages)} messages, {n} signals")
