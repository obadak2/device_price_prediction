package org.example;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;


@RestController
public class DeviceController {
	@Autowired
	private DeviceService deviceService;

	@GetMapping("/get_all_devices")
	public Iterable<Device> getAllDevices() {
		return deviceService.getAllDevices();
	}

	@GetMapping("/{id}")
	public ResponseEntity<Device> getDeviceById(@PathVariable Long id) {
		Device device = deviceService.getDeviceById(id);
		System.out.println("printing the device info"+ResponseEntity.ok(device));
		if (device != null) {
			return ResponseEntity.ok(device);
		} else {
			return ResponseEntity.notFound().build();
		}
	}

	@PostMapping
	public Device saveDevice(@RequestBody Device device) {
		return deviceService.saveDevice(device);
	}

	@PostMapping("/predict/{deviceId}")
	public ResponseEntity<Map> predictPrice(@PathVariable Long deviceId) throws Exception {
		System.out.println("called the predict post mapping");
		Map predictedPrice = deviceService.predictPrice(deviceId);
		return ResponseEntity.ok(predictedPrice);
	}
}
