package org.example;

import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;
import org.springframework.data.repository.query.Param;

import java.util.Optional;

public interface  DeviceRepository extends CrudRepository<Device, Integer> {
    @Query("SELECT d FROM Device d WHERE d.id = :id")
    Optional<Device> findDeviceById(@Param("id") Long id);
}
