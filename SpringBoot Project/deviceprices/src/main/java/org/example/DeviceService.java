package org.example;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
@Service
public class DeviceService {
    @Autowired
    DeviceRepository deviceRepository;
    private final String pythonApiUrl = "http://localhost:8000/predict_price";  // Replace with actual URL


    public Device getDeviceById(Long id) {
        Optional<Device> device = deviceRepository.findDeviceById(id);
        return device.orElse(null);
    }

    public Iterable<Device> getAllDevices() {
        System.out.println("getting all the devices"+deviceRepository.findAll());
        return deviceRepository.findAll();
    }

    public Device saveDevice(Device device) {
        // Save the device object using DeviceRepository
        return deviceRepository.save(device);
    }

    public Map predictPrice(Long deviceId) throws Exception {
        System.out.println("called the predict device service");

        Device device = getDeviceById(deviceId);
        // Prepare request data (Optional):
        Map<String, Object> requestData = new HashMap<>();

//        requestData.put("deviceId", deviceId);  // Replace with actual parameter name
        requestData.put("battery_power", device.getBattery_power());
        requestData.put("blue", device.getBlue());
        requestData.put("clock_speed", device.getClock_speed());
        requestData.put("dual_sim", device.getDual_sim());
        requestData.put("fc", device.getFc());
        requestData.put("four_g", device.getFour_g());
        requestData.put("int_memory", device.getInt_memory());
        requestData.put("m_dep", device.getM_dep());
        requestData.put("mobile_wt", device.getMobile_wt());
        requestData.put("n_cores", device.getN_cores());
        requestData.put("pc", device.getPc());
        requestData.put("px_height", device.getPx_height());
        requestData.put("px_width", device.getPx_width());
        requestData.put("ram", device.getRam());
        requestData.put("sc_h", device.getSc_h());
        requestData.put("sc_w", device.getSc_w());
        requestData.put("talk_time", device.getTalk_time());
        requestData.put("three_g", device.getThree_g());
        requestData.put("touch_screen", device.getTouch_screen());
        requestData.put("wifi", device.getWifi());

        // Make HTTP call using RestTemplate (or another library):
        RestTemplate restTemplate = new RestTemplate();
        ResponseEntity<Map> response = restTemplate.postForEntity(pythonApiUrl, requestData, Map.class);

        System.out.println("The response"+response);
        if (response.getStatusCode() == HttpStatus.OK) {
            return response.getBody();
        } else {
            throw new Exception("Failed to predict price from Python API: " + response.getStatusCodeValue());
        }
    }

}
